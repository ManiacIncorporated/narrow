"""
This script first prunes a model based on attribution scores, then performs
training to hopefully recover performance lost during pruning.
"""

import argparse
import json
import os
# os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
# os.environ['HF_HOME'] = os.environ.get('SCRATCH') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
import gc
from tqdm.auto import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)

# --- Ensure fast tokenization and prevent fork deadlocks ---
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (or containers of tensors) to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    return data


def prepare_data(
    dataset_name: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_samples: int,
    split: str = "train",
    streaming: bool = True,
    skip_samples: int = 0,
):
    """
    Load and tokenize a dataset for pruning or evaluation.

    If the dataset is streamed, you can optionally skip a number of documents,
    allowing you to use one part of the stream for attribution and a different part for evaluation.
    
    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model used to load its tokenizer.
        max_length: Maximum token length.
        batch_size: Batch size to use in the DataLoader.
        num_samples: Number of samples to use from the dataset.
        split: Which split of the dataset to use (e.g. "train", "test").
        streaming: Whether to load the dataset in streaming mode.
        skip_samples: Number of samples to skip from the beginning of the stream.
    
    Returns:
        A DataLoader yielding batches suitable for language modeling.
    """
    if streaming:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        if skip_samples > 0:
            dataset = dataset.skip(skip_samples)
        dataset = dataset.take(num_samples)
        dataset = list(dataset)
    else:
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.select(range(skip_samples, skip_samples + num_samples))

    # Now, after DataLoader workers are created, instantiate tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        prompt = f"Q: {examples['question']}\nA: {examples['answer']}"
        question_part = f"Q: {examples['question']}\nA: "
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
        )
        question_length = len(tokenizer(question_part, add_special_tokens=False)["input_ids"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["question_length"] = question_length
        # Add answer string for BERTScore reference
        tokenized["answer"] = examples["answer"]
        return tokenized

    # Tokenize after DataLoader creation
    if isinstance(dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in dataset]
    else:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    return tokenized_dataset

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on the provided data and compute average loss.
    
    Args:
        model: The language model to evaluate.
        dataloader: DataLoader providing evaluation batches.
        device: The device on which the model and data reside.
    
    Returns:
        A dictionary of evaluation statistics.
    """
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="evaluating..."):
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())
    return {
        "mean_loss": np.mean(losses).item(),
        "std_of_mean": (np.std(losses) / np.sqrt(len(losses))).item(),
        "losses": losses,
    }


def mask_by_gradient_attribution(
    model: nn.Module,
    dataloader: DataLoader,
    neuron_sparsity: float,
    residual_sparsity: float,
    num_attribution_batches: int,
    output_dir: str, 
):
    """
    Prune neurons and residual stream dimensions based on their attribution scores.
    
    Args:
        model: The language model to prune.
        dataloader: DataLoader providing training batches for attribution.
        neuron_sparsity: Fraction of neurons to prune.
        residual_sparsity: Fraction of residual stream dimensions to prune.
        num_attribution_batches: Number of batches to use for computing attribution scores.
        output_dir: Directory to save pruning information.
    """
    model.train()  # Set to train mode to enable gradients

    param_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    num_samples = 0
    for i, batch in enumerate(tqdm(dataloader, desc="computing mean gradients...")):
        if i >= num_attribution_batches:
            break
        model.zero_grad()
        batch = move_to_device(batch, model.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grads[name] += param.grad.abs().detach()
        num_samples += batch['input_ids'].size(0)
        model.zero_grad()
    for name in param_grads:
        if num_samples > 0:
            param_grads[name] /= num_samples

    neuron_scores = {}
    for layeri, layer in enumerate(model.model.layers):
        gp_grad = param_grads[f"model.layers.{layeri}.mlp.gate_proj.weight"]
        up_grad = param_grads[f"model.layers.{layeri}.mlp.up_proj.weight"]
        dp_grad = param_grads[f"model.layers.{layeri}.mlp.down_proj.weight"]
        gp = layer.mlp.gate_proj.weight
        up = layer.mlp.up_proj.weight
        dp = layer.mlp.down_proj.weight
        neuron_scores[layeri] = torch.sum(
            (gp_grad * -gp) + 
            (up_grad * -up) + 
            (dp_grad.T * -dp.T), 
            dim=1
        ).abs().tolist()
    
    d_model = model.config.hidden_size
    device = model.model.embed_tokens.weight.device
    dtype = model.model.embed_tokens.weight.dtype
    residual_scores = torch.zeros(d_model, device=device, dtype=dtype)
    residual_scores += (param_grads[f"model.embed_tokens.weight"] * -model.model.embed_tokens.weight).sum(dim=0)
    for layeri, layer in enumerate(model.model.layers):
        residual_scores += param_grads[f"model.layers.{layeri}.input_layernorm.weight"] * -layer.input_layernorm.weight
        residual_scores += param_grads[f"model.layers.{layeri}.post_attention_layernorm.weight"] * -layer.post_attention_layernorm.weight
        residual_scores += (param_grads[f"model.layers.{layeri}.mlp.gate_proj.weight"] * -layer.mlp.gate_proj.weight).sum(dim=0)
        residual_scores += (param_grads[f"model.layers.{layeri}.mlp.up_proj.weight"] * -layer.mlp.up_proj.weight).sum(dim=0)
        residual_scores += (param_grads[f"model.layers.{layeri}.mlp.down_proj.weight"] * -layer.mlp.down_proj.weight).sum(dim=1)
        residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.q_proj.weight"] * -layer.self_attn.q_proj.weight).sum(dim=0)
        residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.k_proj.weight"] * -layer.self_attn.k_proj.weight).sum(dim=0)
        residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.v_proj.weight"] * -layer.self_attn.v_proj.weight).sum(dim=0)
        residual_scores += (param_grads[f"model.layers.{layeri}.self_attn.o_proj.weight"] * -layer.self_attn.o_proj.weight).sum(dim=1)
    residual_scores += param_grads[f"model.norm.weight"] * -model.model.norm.weight
    residual_scores = residual_scores.abs().tolist()

    mask = {name: torch.ones_like(param) for name, param in model.named_parameters()}

    neuron_score_tuples = [
        (layeri, neuroni, neuron_scores[layeri][neuroni]) 
        for layeri in neuron_scores for neuroni in range(len(neuron_scores[layeri]))
    ]
    neuron_score_tuples.sort(key=lambda x: x[2])  # Sort by score (ascending)
    n_neurons = sum(layer.mlp.gate_proj.out_features for layer in model.model.layers)
    neurons_to_prune_count = int(n_neurons * neuron_sparsity)
    pruned_neurons = []
    for i in range(min(neurons_to_prune_count, len(neuron_score_tuples))):
        layeri, neuroni, _ = neuron_score_tuples[i]
        pruned_neurons.append((layeri, neuroni))
    for layeri, neuroni in pruned_neurons:
        mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][neuroni, :] = 0
        mask[f"model.layers.{layeri}.mlp.up_proj.weight"][neuroni, :] = 0
        mask[f"model.layers.{layeri}.mlp.down_proj.weight"][:, neuroni] = 0
    
    residual_score_tuples = [(i, residual_scores[i]) for i in range(len(residual_scores))]
    residual_score_tuples.sort(key=lambda x: x[1])  # Sort by score (ascending)
    n_residuals = model.config.hidden_size
    residuals_to_prune_count = int(n_residuals * residual_sparsity)
    pruned_residuals = []
    for i in range(min(residuals_to_prune_count, len(residual_score_tuples))):
        dim_idx, _ = residual_score_tuples[i]
        pruned_residuals.append(dim_idx)
    mask[f"model.embed_tokens.weight"][:, pruned_residuals] = 0
    for layeri, layer in enumerate(model.model.layers):
        mask[f"model.layers.{layeri}.input_layernorm.weight"][pruned_residuals] = 0
        mask[f"model.layers.{layeri}.post_attention_layernorm.weight"][pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.gate_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.up_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.mlp.down_proj.weight"][pruned_residuals, :] = 0
        mask[f"model.layers.{layeri}.self_attn.q_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.k_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.v_proj.weight"][:, pruned_residuals] = 0
        mask[f"model.layers.{layeri}.self_attn.o_proj.weight"][pruned_residuals, :] = 0
    mask[f"model.norm.weight"][pruned_residuals] = 0
    
    # Apply mask to model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
    
    stats = {
        "pruned_neurons": pruned_neurons,
        "pruned_residuals": pruned_residuals,
        "neuron_scores": neuron_scores,
        "residual_scores": residual_scores,
        "total_neurons": n_neurons,
        "total_residuals": n_residuals,
        "total_neurons_pruned": len(pruned_neurons),
        "total_residuals_pruned": len(pruned_residuals),
    }

    # Print statistics for debugging
    print("\n=== Pruning Statistics ===")
    print(f"Total neurons: {n_neurons}")
    print(f"Total residuals: {n_residuals}")
    print(f"Neurons pruned: {len(pruned_neurons)} / {n_neurons} ({len(pruned_neurons)/n_neurons:.2%})")
    print(f"Residuals pruned: {len(pruned_residuals)} / {n_residuals} ({len(pruned_residuals)/n_residuals:.2%})")
    
    # Print some of the pruned indices for verification
    if pruned_neurons:
        print(f"\nSample of pruned neurons: {pruned_neurons[:5]}{'...' if len(pruned_neurons) > 5 else ''}")
    if pruned_residuals:
        print(f"Sample of pruned residuals: {pruned_residuals[:5]}{'...' if len(pruned_residuals) > 5 else ''}")
    
    # Print some attribution score statistics
    if neuron_scores:
        # Convert dictionary of lists to a flat numpy array
        neuron_scores_array = np.concatenate([np.array(scores) for scores in neuron_scores.values()])
        print(f"\nNeuron attribution scores - min: {neuron_scores_array.min():.6f}, max: {neuron_scores_array.max():.6f}, mean: {neuron_scores_array.mean():.6f}")
    
    if residual_scores:
        residual_scores_array = np.array(residual_scores)
        print(f"Residual attribution scores - min: {residual_scores_array.min():.6f}, max: {residual_scores_array.max():.6f}, mean: {residual_scores_array.mean():.6f}")
    print("===========================\n")
    
    return mask, stats


class MaskedTrainer(Trainer):
    """
    Custom Trainer that applies a mask to model parameters every mask_steps steps.
    This ensures pruned neurons remain pruned during training.
    Optionally adds a differentiable embedding similarity loss (BERTScore-style) to the standard cross-entropy loss.
    n.b. that embedding_loss_alpha scales the embedding loss.
    """
    def __init__(self, mask=None, mask_steps=1, embedding_loss_alpha=0.0, print_debug: int = None, **kwargs):
        """
        Initialize the MaskedTrainer.
        Args:
            mask: Dictionary mapping parameter names to binary masks
            mask_steps: Apply mask every this many steps
            embedding_loss_alpha: Weight of embedding similarity loss (0 = off)
            **kwargs: Arguments to pass to the parent Trainer
        """
        super().__init__(**kwargs)
        self.mask = mask
        self.mask_steps = mask_steps
        self.embedding_loss_alpha = embedding_loss_alpha
        self.print_debug = print_debug

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard LM loss
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Embedding similarity loss (true BERTScore-style using answer string)
        emb_loss = 0.0
        if self.embedding_loss_alpha > 0.0:
            input_ids = inputs['input_ids']  # (batch, seq)
            labels = inputs['labels']        # (batch, seq)
            attention_mask = inputs.get('attention_mask', None)
            answer_mask = (labels != -100)
            pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0

            # Reference: tokenize answer string, pad to max length in batch
            answer_texts = inputs['answer']  # List[str], field from collator/dataset
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else self.processing_class
            
            with torch.no_grad():

                ref_tokenized = tokenizer(answer_texts, return_tensors='pt', padding=True, truncation=True)
                ref_input_ids = ref_tokenized['input_ids'].to(input_ids.device)
                ref_attention_mask = ref_tokenized['attention_mask'].to(input_ids.device)
                ref_outputs = model(input_ids=ref_input_ids, attention_mask=ref_attention_mask, output_hidden_states=True)
                
                # Get reference embeddings immediately and detach
                ref_hidden = ref_outputs.hidden_states[-1].detach()
                ref_emb_list = []
                for i in range(ref_input_ids.size(0)):
                    ref_ans_idx = ref_input_ids[i] != pad_token_id
                    if ref_ans_idx.sum() == 0:
                        ref_emb_list.append(ref_hidden[i].mean(dim=0))
                    else:
                        ref_emb_list.append(ref_hidden[i][ref_ans_idx].mean(dim=0))
                ref_emb = torch.stack(ref_emb_list, dim=0)
                
                # Clean up reference computation immediately
                del ref_outputs, ref_hidden, ref_tokenized, ref_input_ids, ref_attention_mask, ref_emb_list
            gc.collect() # ref_outputs stores tensors with attached computation graphs, so delete
            
            # Get predicted answer embeddings (keep gradients for this part)
            pred_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            pred_hidden = pred_outputs.hidden_states[-1].detach()
            
            # Compute predicted embeddings
            pred_emb_list = []
            for i in range(input_ids.size(0)):
                ans_idx = answer_mask[i].nonzero(as_tuple=True)[0]
                if len(ans_idx) == 0:
                    pred_emb_list.append(pred_hidden[i].mean(dim=0))
                else:
                    pred_emb_list.append(pred_hidden[i][ans_idx].mean(dim=0))
            pred_emb = torch.stack(pred_emb_list, dim=0)
            
            # Compute cosine similarity and loss
            cosine_sim = torch.nn.functional.cosine_similarity(pred_emb, ref_emb, dim=-1)
            emb_loss = 1 - cosine_sim.mean()
            loss = loss + self.embedding_loss_alpha * emb_loss
            
            # Clean up embedding computation
            del pred_outputs, pred_hidden, pred_emb, ref_emb, cosine_sim, pred_emb_list
            gc.collect()
        
        # Clean up main outputs
        if not return_outputs:
            del outputs
            
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """Override training_step to apply masks periodically and add embedding loss if enabled. Uses CUDA AMP if available."""
        use_amp = torch.cuda.is_available()

        # More efficient tensor counting - only count every 10 steps to reduce overhead
        if self.print_debug is not None:
            if self.state.global_step % self.print_debug == 0:
                num_tensors = sum(1 for obj in gc.get_objects() if torch.is_tensor(obj))
                print(f"Step {self.state.global_step}: {num_tensors} tensors, {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB allocated")

        model.train()
        inputs = self._prepare_inputs(inputs)
        
        if use_amp:
            from torch.amp import autocast
            with autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
            
        # Apply mask every mask_steps
        if self.state.global_step % self.mask_steps == 0 and self.mask is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.mask:
                        param.data *= self.mask[name]
        
        del inputs
        torch.cuda.empty_cache()
        loss.backward()

        return loss.detach()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for both pruning and training phases.
    """
    parser = argparse.ArgumentParser(
        description="Prune a model based on attribution scores, then train to recover performance."
    )
    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Pretrained model name or path.")
    parser.add_argument("--dataset_name", type=str, default="BoltMonkey/psychology-question-answer",
                        help="Dataset name for pruning and training.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--accumulations", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--streaming", action="store_true",
                        help="Load the dataset in streaming mode.")
    parser.add_argument("--output_dir", type=str, default="./pruned_trained_models",
                        help="Directory to save the pruned and trained model.")

    # Pruning parameters
    parser.add_argument("--neuron_sparsity", type=float, default=0.8,
                        help="Fraction of neurons to prune.")
    parser.add_argument("--residual_sparsity", type=float, default=0.5,
                        help="Fraction of residual stream dimensions to prune.")
    parser.add_argument("--prune_samples", type=int, default=1000,
                        help="Number of samples to use for pruning data.")
    parser.add_argument("--prune_skip", type=int, default=0,
                        help="Number of samples to skip for pruning (if streaming).")

    # Training parameters
    # parser.add_argument("--train_samples", type=int, default=,
    #                     help="Number of samples to use for training data.")
    parser.add_argument("--train_skip", type=int, default=0,
                        help="Number of samples to skip for training (if streaming).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total number of training steps to run. -1 means use num_train_epochs.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for training.")
    parser.add_argument("--mask_steps", type=int, default=1,
                        help="Apply mask every this many steps during training.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--limit_checkpoints", type=int, default=3,
                        help="Limit the number of checkpoints saved. Set to -1 for unlimited")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for learning rate scheduler.")

    # Evaluation parameters
    parser.add_argument("--eval", action="store_true",
                        help="Whether to perform evaluation after pruning and training.")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Number of samples to use for evaluation.")
    parser.add_argument("--eval_skip", type=int, default=0,
                        help="Number of samples to skip for evaluation (if streaming).")

    return parser.parse_args()

def main() -> None:
    
    # args = parse_args()

    ##### DT 7/14: Manually define parameters; ignore argparse as we are not slurming
    # Base model to use
    model_name = "NousResearch/Llama-3.2-1B"
    # Define specific combinations of (neuron_sparsity, residual_sparsity)
    sparsity_configs = [
        # (0.5, 0.2),
        (0.5, 0.5),
        (0.8, 0.5),
        (0.9, 0.5),
        (0.95, 0.5),
        (0.8, 0.8),
        (0.9, 0.9),
    ]
    # Common training parameters
    training_params = {
        "max_length": 512, # 1024
        "batch_size": 8,
        "accumulations": 4,
        "prune_samples": 1024,
        "train_skip": 1024,
        "max_steps": 7500, # 20000
        "lr": 1e-4,
        "mask_steps": 1,
        "eval_steps": 500, # 500
        "save_steps": 2500, # 2500
        "limit_checkpoints": 3,
        "logging_steps": 1000,
        "debug_steps": None, # Int, if not None, print debug (memory) info every debug_steps
        "warmup_steps": 500, # 1000
        # Phase-specific step counts
        "phase1_steps": 2500, # 10000
        "phase2_steps": 5000, # 10000
    }

    # Create logs directory
    os.makedirs("./psych-llm/logs", exist_ok=True)
    
    # Launch jobs for each sparsity configuration
    neuron_sparsity, residual_sparsity = sparsity_configs[0]
    # Create a unique name for this configuration
    config_name = f"n{neuron_sparsity:.2f}_r{residual_sparsity:.2f}"
    
    # Create output directory
    output_dir = f'./psych-llm/{config_name}'
    os.makedirs(output_dir)
    
    # Format the script with all parameters
    args = {
        'model_name': model_name,
        'dataset_name': 'BoltMonkey/psychology-question-answer',
        'neuron_sparsity': neuron_sparsity,
        'residual_sparsity': residual_sparsity,
        'output_dir': output_dir,
        'max_length': training_params["max_length"],
        'batch_size': training_params["batch_size"],
        'accumulations': training_params["accumulations"],
        'prune_samples': training_params["prune_samples"],
        'train_skip': training_params["train_skip"],
        'max_steps': training_params["max_steps"],
        'lr': training_params["lr"],
        'mask_steps': training_params["mask_steps"],
        'eval_steps': training_params["eval_steps"],
        'save_steps': training_params["save_steps"],
        'limit_checkpoints': training_params["limit_checkpoints"],
        'logging_steps': training_params["logging_steps"],
        'warmup_steps': training_params["warmup_steps"],
        'streaming': True,
        'prune_skip': 0,
        'eval': True,
        'eval_skip': 2,
        'eval_samples': 512
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args['output_dir'], exist_ok=True)
    
    print(f"Loading model: {args['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        args['model_name'],
        torch_dtype=torch.float32,
        device_map=str(device)
    )
 
    # ===== STEP 1: Lasso regularization with embedding similarity loss =====
    
    # Load tokenizer for training data preparation
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and tokenize training data using prepare_data (returns DataLoader)
    print("Preparing training data...")
    train_dataset = prepare_data(
        args['dataset_name'],
        args['model_name'],
        args['max_length'],
        args['batch_size'],
        args['batch_size'] * args['max_steps'] * args['accumulations'],
        split="train",
        streaming=args['streaming'],
        skip_samples=args['train_skip'],
    )
    
    # Load evaluation data if needed
    if args['eval']:
        print("Preparing evaluation data...")
        eval_dataset = prepare_data(
            args['dataset_name'],
            args['model_name'],
            args['max_length'],
            args['batch_size'],
            args['eval_samples'],
            split="train",
            streaming=args['streaming'],
            skip_samples=args['eval_skip'],
        )
    else:
        eval_dataset = None
    
    # Data collator for training
    from qa_data_collator import QADataCollator
    data_collator = QADataCollator()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args['output_dir'],
        max_steps=args['max_steps'] if args['max_steps'] > 0 else -1,
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        gradient_accumulation_steps=args['accumulations'],
        learning_rate=args['lr'],
        warmup_steps=args['warmup_steps'],
        logging_dir=os.path.join(args['output_dir'], "logs"),
        logging_steps=args['logging_steps'],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args['eval_steps'] if eval_dataset else None,
        save_strategy="steps",
        save_steps=args['save_steps'],
        save_total_limit=args['limit_checkpoints'],
        load_best_model_at_end=eval_dataset is not None,
        bf16=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        weight_decay=0.00,
        dataloader_num_workers=1, # With this dataset num_shards=1
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        gradient_checkpointing=True,
        prediction_loss_only=True
        # fp16=False,
    )
    
    # Initialize the MaskedTrainer with no mask
    trainer = MaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        mask=None,
        mask_steps=args['mask_steps'],
        embedding_loss_alpha=0.2,  # Enable embedding similarity loss for phase 1
        print_debug=training_params['debug_steps'],
    )

    torch.cuda.empty_cache()
    
    # === Phase 1: Fine-tune with group lasso regularization and cosine similarity ===
    print("Phase 1: Fine-tuning with L1 regularization (lasso)...")
    from l1_regularizer import l1_of_l2_of_mlps
    lasso_lambda = 5e-3 # You may want to tune this
    phase1_steps = training_params["phase1_steps"]
    trainer.args.max_steps = phase1_steps
    original_compute_loss = trainer.compute_loss
    def compute_loss_with_l1(model, inputs, return_outputs=False):
        loss = original_compute_loss(model, inputs, return_outputs)
        l1_loss = l1_of_l2_of_mlps(model)
        if isinstance(loss, tuple):
            loss_val, outputs = loss
            return loss_val + lasso_lambda * l1_loss, outputs
        return loss + lasso_lambda * l1_loss
    trainer.compute_loss = compute_loss_with_l1
    trainer.train()
    # Save intermediate model
    trainer.save_model(os.path.join(args['output_dir'], "phase1_lasso_model"))
    print("Phase 1 complete. Model saved.\n")

    # ===== STEP 2: Pruning based on attribution scores after regularizing weights =====
    # Continue fine-tuning with an embedding similarity score to naively ensure alignment

    # Reload model from phase 1 (after lasso regularization)
    model = AutoModelForCausalLM.from_pretrained(os.path.join(args['output_dir'], "phase1_lasso_model"), torch_dtype=torch.bfloat16, device_map=str(device))
 
    # Load pruning data
    print("Preparing pruning data...")
    pruning_dataset = prepare_data(
        dataset_name=args['dataset_name'],
        model_name=args['model_name'],
        max_length=args['max_length'],
        batch_size=args['batch_size'],
        num_samples=args['prune_samples'],
        split="train",
        streaming=args['streaming'],
        skip_samples=args['prune_skip']
    )
    
    pruning_dataloader = DataLoader(pruning_dataset, batch_size=args['batch_size'], collate_fn=data_collator)

    # Create mask based on attribution scores
    print("Creating pruning mask based on attribution scores...")
    num_attribution_batches = args['prune_samples'] // args['batch_size']
    mask, pruning_stats = mask_by_gradient_attribution(
        model=model,
        dataloader=pruning_dataloader,
        neuron_sparsity=args['neuron_sparsity'],
        residual_sparsity=args['residual_sparsity'],
        num_attribution_batches=num_attribution_batches,
        output_dir=args['output_dir']
    )
    
    # Save initial pruning statistics
    pruning_stats_file = os.path.join(args['output_dir'], "pruning_stats.json")
    with open(pruning_stats_file, "w") as f:
        json.dump(pruning_stats, f, indent=4)
    print(f"Pruning statistics saved to {pruning_stats_file}")
    # save mask as a torch file
    mask_file = os.path.join(args['output_dir'], "pruning_mask.pt")
    torch.save(mask, mask_file)

    # Setup trainer for phase 2 (pruned model)
    trainer = MaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        mask=mask,
        mask_steps=args['mask_steps'],
        embedding_loss_alpha=getattr(trainer, 'embedding_loss_alpha', 0.2),
        print_debug=training_params['debug_steps'],
    )
    phase2_steps = training_params["phase2_steps"]
    trainer.args.max_steps = phase1_steps + phase2_steps  # total steps
    trainer.state.global_step = phase1_steps  # continue from step 10k
    trainer.compute_loss = original_compute_loss  # remove L1 reg
    trainer.train(resume_from_checkpoint=None)
    # Save the final model
    trainer.save_model(os.path.join(args['output_dir'], "final_model"))
    tokenizer.save_pretrained(os.path.join(args['output_dir'], "final_model"))
    print(f"Final model saved to {os.path.join(args['output_dir'], 'final_model')}")

    # === Compression statistics ===
    print("Computing compression statistics...")
    from compression_stats import model_sparsity_stats, model_num_parameters, model_num_bytes
    # Load original pretrained model for baseline
    orig_model = AutoModelForCausalLM.from_pretrained(args['model_name'], torch_dtype=torch.bfloat16, device_map=str(device))
    orig_params = model_num_parameters(orig_model)
    orig_bytes = model_num_bytes(orig_model)
    pruned_params = model_num_parameters(model)
    pruned_bytes = model_num_bytes(model)
    sparsity = model_sparsity_stats(model)
    compression_stats = {
        'orig_num_parameters': orig_params,
        'orig_num_bytes': orig_bytes,
        'pruned_num_parameters': pruned_params,
        'pruned_num_bytes': pruned_bytes,
        'memory_compression_ratio': pruned_bytes / orig_bytes,
        'parameter_compression_ratio': pruned_params / orig_params,
        'fraction_removed': sparsity['fraction_removed'],
        'layerwise_sparsity': sparsity['layerwise'],
    }
    stats_file = os.path.join(args['output_dir'], "compression_stats.json")
    with open(stats_file, "w") as f:
        json.dump(compression_stats, f, indent=4)
    print(f"Compression statistics saved to {stats_file}")
    
    # Evaluate the final model if requested
    if args['eval']:
        print("Evaluating final model...")
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=args['batch_size'], 
            collate_fn=data_collator
        )
        eval_stats = evaluate_model(model, eval_dataloader, device)
        
        # Save evaluation results
        eval_file = os.path.join(args['output_dir'], "final_evaluation_results.json")
        with open(eval_file, "w") as f:
            json.dump(eval_stats, f, indent=4)
        print(f"Final evaluation results saved to {eval_file}")


if __name__ == "__main__":
    main()







