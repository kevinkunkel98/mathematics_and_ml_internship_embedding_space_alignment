import torch
import torch.nn as nn

class CKALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        X, Y: (B, D) - Activations aus beiden Modellen
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Sample count mismatch für CKA Loss")

        # WICHTIG: Da Sprachmodelle oft in BFloat16 laufen, konvertieren wir hier 
        # für die Matrix-Multiplikation kurz in Float32, um numerisches Unter-/Überlaufen zu verhindern.
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)

        # Deine saubere Zentrierung beibehalten
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        K = X @ X.T
        L = Y @ Y.T

        # Elementweise Multiplikation + Summe ist identisch zu der Spur (Trace) von K @ L,
        # aber dramatisch schneller und speichereffizienter für das Autograd-System.
        hsic = torch.sum(K * L)
        norm = torch.norm(K, p="fro") * torch.norm(L, p="fro")

        cka = hsic / (norm + 1e-8)  # Epsilon gegen Division durch 0

        # Wir minimieren (1 - CKA)
        return 1.0 - cka