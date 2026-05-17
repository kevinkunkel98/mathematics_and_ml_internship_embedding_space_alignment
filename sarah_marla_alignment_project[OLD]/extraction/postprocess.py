# extraction/postprocess.py
"""
    def vit_postprocess(x):
    # exclude CLS token
    return x[:, 1:, :].mean(dim=1)  # (B, D)

    Best for:
    spatial equivalence with CNN GAP


    Hook logic (inside extraction)
    activations = {}

    def hook(name):
        def fn(_, __, output):
            activations[name] = postprocess_fn(output).cpu()
        return fn
"""

def cnn_postprocess(x):
    # (B, C, H, W) → (B, C·H·W)
    return x.flatten(start_dim=1)

def vit_postprocess(x):
    """
    x: (B, N_tokens, D) is not directly comparable to CNN maps unless you normalize correctly.

    Best for:
        semantic alignment
        layer-depth comparisons    
    """
    # (B, N, D) → (B, D)
    return x[:, 0, :]