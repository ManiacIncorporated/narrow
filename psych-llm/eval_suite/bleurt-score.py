import json
import os
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
import pickle
from tqdm.auto import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GenerationConfig
) 
# n.b.: please downgrade transformers to transformers==4.49.0 or 4.50.0. 
# bleurt_pytorch is a port from 2023 with .pt model states, not .safetensors.
# as a result, torch shoots you if you try to load from_pretrained ;
# long-term #TODO here is to use bleurt_pytorch lib to train our own BLEURT and save as .safetensors.

from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

@torch.no_grad()
def main(data: str = None, model_path: str = None, scores_path: str = None) -> None:

    # Load inference model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=str(device))
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', device_map=str(device))
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig(max_new_tokens=100, do_sample=True, top_k=50, bos_token_id=128000, eos_token_id=128001)

    # Load scoring model
    scoring_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
    scoring_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20', torch_dtype=torch.float32, device_map=str(device))
    scoring_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20', torch_dtype=torch.float32, device_map=str(device))

    # load evaluation data
    if data:
        eval_dataset = load_dataset(data, split="train")
    else:
        raise Exception("No evaluation dataset specified.")

    # begin scoring loop
    model.eval()
    scoring_model.eval()
    BATCH_SIZE = 100
    NUM_BATCH = 50 # len(eval_dataset['question'])//BATCH_SIZE
    scores = []

    for i in tqdm(range(NUM_BATCH), desc='scoring'):
        torch.cuda.empty_cache()
        batch_input = eval_dataset.select(range(i*BATCH_SIZE,(i+1)*BATCH_SIZE))
        raw_prompts = [q for q in batch_input['question']]
        tokens_in = tokenizer(raw_prompts, padding=True, return_tensors='pt').to(device)
        tokens_out = model.generate(**tokens_in, generation_config = generation_config)
        raw_model_response = tokenizer.batch_decode(tokens_out, skip_special_tokens=True)
        del raw_prompts; del tokens_in; del tokens_out
   
        # Re-tokenize with pretrained BLEURT and score
        raw_answers = [a for a in batch_input['answer']]
        score_inputs = scoring_tokenizer(raw_answers, raw_model_response, padding='longest', return_tensors='pt').to(device)
        scores = scores + scoring_model(**score_inputs).logits.flatten().tolist()
        del batch_input; raw_model_response; del raw_answers; del score_inputs

        gc.collect()
        
    print(f'Batch scoring done on {NUM_BATCH} batch(es) of size {BATCH_SIZE}.')
    with open(scores_path, 'wb') as f:
        pickle.dump(scores, f)

if __name__ == "__main__":
    dataset_name = "BoltMonkey/psychology-question-answer"
    model_path = "./../n0.50_r0.50/final_model/"
    scores_path = "./scores.pkl"
    main(data=dataset_name, model_path=model_path, scores_path=scores_path)
