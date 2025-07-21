from typing import List, Dict, Any
import torch

class QADataCollator:
    """
    Data collator that masks loss on question tokens and computes loss only on answer tokens.
    Assumes each input is a dict with 'input_ids', 'labels', and 'question_length'.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for k in features[0]:
            batch[k] = [f[k] for f in features]
        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in batch['input_ids']], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in batch['labels']], batch_first=True, padding_value=-100
        )
        # Mask out question tokens in labels
        for i, qlen in enumerate(batch['question_length']):
            labels[i, :qlen] = -100
        # Return answer string for BERTScore reference
        return {'input_ids': input_ids, 'labels': labels, 'answer': batch['answer']}
