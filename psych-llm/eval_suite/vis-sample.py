import json
import os
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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

def main() -> None:

    model_name = "NousResearch/Llama-3.2-1B"
    local_state_path = "./../n0.50_r0.50/final_model/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(local_state_path, torch_dtype=torch.float32, device_map=str(device))
    tokenizer = AutoTokenizer.from_pretrained(local_state_path, device_mp=str(device))

    print(model)
    
    user_in = input(">>>")
    generation_config = GenerationConfig(max_new_tokens=100, do_sample=True, top_k=50, bos_token_id=128000, eos_token_id=128001)
    while user_in not in ["quit"," q", "Quit", "Q"]:
        tokens_in = tokenizer(user_in, return_tensors="pt").to(device)
        tokens_out = model.generate(**tokens_in, generation_config = generation_config)
        model_out = tokenizer.decode(tokens_out[0], skip_special_tokens=True)
        # Extract only the answer after 'A:'
        def extract_answer(text):
            idx = text.find('A:')
            if idx == -1:
                return text.strip()
            return text[idx+2:].strip()
        answer = extract_answer(model_out)
        print(answer+'\n')
        user_in = input(">>>")

if __name__ == "__main__":
    main()
