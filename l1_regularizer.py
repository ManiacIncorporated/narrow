import torch

def l1_of_l2_of_mlps(model):
    """
    Compute L1 norm of L2 norms of MLP weights (gate, up, down proj) for all layers.
    Used as a sparsity-inducing regularizer for Llama-style models.
    """
    l1_sum = 0.0
    for name, param in model.named_parameters():
        if any(x in name for x in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]):
            # L2 norm over output dim, then L1 over all outputs
            l2 = torch.norm(param, p=2, dim=1)
            l1 = torch.norm(l2, p=1)
            l1_sum = l1_sum + l1
    return l1_sum
