import torch

def count_nonzero_and_total(tensor):
    nz = torch.count_nonzero(tensor).item()
    total = tensor.numel()
    return nz, total

def model_sparsity_stats(model):
    """
    Returns dict with neuron and connection sparsity for all linear layers.
    """
    stats = {
        'total_params': 0,
        'nonzero_params': 0,
        'layerwise': {}
    }
    for name, param in model.named_parameters():
        if param.ndim == 2:  # Linear weights
            nz, total = count_nonzero_and_total(param.data)
            stats['total_params'] += total
            stats['nonzero_params'] += nz
            stats['layerwise'][name] = {'nonzero': nz, 'total': total, 'sparsity': 1 - nz/total}
    stats['fraction_removed'] = 1 - stats['nonzero_params'] / stats['total_params']
    return stats


def model_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_num_bytes(model):
    # Returns total parameter memory in bytes
    return sum(p.numel() * p.element_size() for p in model.parameters())
