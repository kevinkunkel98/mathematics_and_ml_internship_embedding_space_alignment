"""
Fine-tune ResNet-18 and ViT-B/16 on CIFAR-10, then extract:
  - Layer-wise activations for a test subset (for CKA)
  - GradCAM heatmaps for a small image sample

Output:
  data/vision/resnet18.h5
  data/vision/vit_b16.h5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from scripts.io import save_vision_data

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_N_CLASSES = 10
_N_CAM_SAMPLES = 40


def _get_loaders(batch_size: int, n_train: int, n_test: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    raw_tf = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(
        "data/cifar10", train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        "data/cifar10", train=False, download=True, transform=test_tf
    )
    raw_ds = datasets.CIFAR10(
        "data/cifar10", train=False, download=False, transform=raw_tf
    )

    train_loader = DataLoader(
        Subset(train_ds, range(n_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        Subset(test_ds, range(n_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    cam_loader = DataLoader(
        Subset(raw_ds, range(_N_CAM_SAMPLES)), batch_size=_N_CAM_SAMPLES, shuffle=False
    )
    return train_loader, test_loader, cam_loader


def _train(
    model: nn.Module, loader: DataLoader, epochs: int, device: torch.device
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        total, correct = 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        print(f"  epoch {epoch}/{epochs}  train acc: {correct / total:.3f}")


# ── ResNet-18 ────────────────────────────────────────────────────────────────


def _build_resnet(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, _N_CLASSES)
    return model.to(device)


def _extract_resnet_activations(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extract activations after each of the 4 ResNet layer groups + stem."""
    hooks, buffers = [], {}

    def _hook(name):
        def fn(_, __, output):
            buffers[name] = output.detach().cpu()

        return fn

    # stem (after maxpool), 4 layer groups, avgpool
    tap_points = [
        ("stem", model.maxpool),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("avgpool", model.avgpool),
    ]
    for name, module in tap_points:
        hooks.append(module.register_forward_hook(_hook(name)))

    model.eval()
    all_acts = {i: [] for i in range(len(tap_points))}
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            _ = model(X.to(device))
            for i, (name, _) in enumerate(tap_points):
                feat = buffers[name]
                # spatial features: global-average-pool to (batch, dim)
                if feat.dim() == 4:
                    feat = feat.mean(dim=[2, 3])
                elif feat.dim() == 3:
                    feat = feat[:, 0, :]  # CLS token for ViT-style
                all_acts[i].append(feat.numpy())
            all_labels.extend(y.numpy())

    for h in hooks:
        h.remove()

    activations = {i: np.concatenate(v, axis=0) for i, v in all_acts.items()}
    return activations, np.array(all_labels, dtype=np.int8)


def _gradcam_resnet(
    model: nn.Module,
    images_norm: torch.Tensor,
    targets: list[int],
    device: torch.device,
) -> np.ndarray:
    """GradCAM using layer3 (2×2 spatial maps for 32×32 CIFAR-10 input).

    layer4 collapses to 1×1 for CIFAR-10 — not useful for spatial CAMs.
    """
    model.eval()
    cams = []

    feat_maps = {}
    grads = {}

    def fwd_hook(_, __, output):
        feat_maps["layer3"] = output

    def bwd_hook(_, __, grad_output):
        grads["layer3"] = grad_output[0]

    fh = model.layer3.register_forward_hook(fwd_hook)
    bh = model.layer3.register_full_backward_hook(bwd_hook)

    for i, img in enumerate(images_norm):
        img = img.unsqueeze(0).to(device).requires_grad_(True)
        logits = model(img)
        model.zero_grad()
        logits[0, targets[i]].backward()

        weights = grads["layer3"].mean(dim=[2, 3], keepdim=True)
        cam = (weights * feat_maps["layer3"]).sum(dim=1).squeeze()
        cam = torch.relu(cam).detach().cpu().numpy()
        if cam.ndim == 0:
            cam = np.ones((2, 2), dtype=np.float32) * float(cam)
        cam_resized = _resize_cam(cam, 32)
        cams.append(cam_resized)

    fh.remove()
    bh.remove()
    return np.stack(cams)


# ── ViT-B/16 ─────────────────────────────────────────────────────────────────


def _build_vit(device: torch.device) -> nn.Module:
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(768, _N_CLASSES)
    return model.to(device)


def _extract_vit_activations(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extract CLS token from each of the 12 transformer encoder blocks."""
    hooks, buffers = [], {}

    def _hook(i):
        def fn(_, __, output):
            buffers[i] = output[:, 0, :].detach().cpu()

        return fn

    for i, block in enumerate(model.encoder.layers):
        hooks.append(block.register_forward_hook(_hook(i)))

    # Also tap the patch embedding output (layer 0 = embedding)
    patch_buf = {}

    def _emb_hook(_, __, output):
        patch_buf[0] = output[:, 0, :].detach().cpu()

    hooks.append(
        model.encoder.register_forward_hook(lambda m, i, o: None)
    )  # placeholder
    emb_h = model.conv_proj.register_forward_hook(
        lambda m, inp, out: patch_buf.update(
            {-1: out.flatten(2).transpose(1, 2)[:, 0, :].detach().cpu()}
        )
    )

    model.eval()
    n_blocks = len(model.encoder.layers)
    all_acts = {i: [] for i in range(-1, n_blocks)}
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            _ = model(X.to(device))
            all_acts[-1].append(patch_buf[-1].numpy())
            for i in range(n_blocks):
                all_acts[i].append(buffers[i].numpy())
            all_labels.extend(y.numpy())

    for h in hooks:
        h.remove()
    emb_h.remove()

    # Re-index from 0
    activations = {}
    for j, k in enumerate(sorted(all_acts.keys())):
        activations[j] = np.concatenate(all_acts[k], axis=0)
    return activations, np.array(all_labels, dtype=np.int8)


def _attention_cam_vit(
    model: nn.Module, images_norm: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Attention rollout over all 12 encoder blocks, CLS→patch row, resized to 32×32.

    Uses register_forward_pre_hook to force need_weights=True since EncoderBlock
    calls self_attention(..., need_weights=False) by default.
    """
    model.eval()
    n_blocks = len(model.encoder.layers)
    # ViT-B/16 with 224×224 input → 14×14 = 196 patches + 1 CLS = 197 tokens
    grid = 14

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    for i, block in enumerate(model.encoder.layers):

        def _pre(mod, args, kwargs, _i=i):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return args, kwargs

        def _post(mod, inp, out, _i=i):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                captured[_i] = out[1].detach().cpu()  # (1, seq, seq)

        hooks.append(
            block.self_attention.register_forward_pre_hook(_pre, with_kwargs=True)
        )
        hooks.append(block.self_attention.register_forward_hook(_post))

    cams = []
    with torch.no_grad():
        for img in images_norm:
            captured.clear()
            _ = model(img.unsqueeze(0).to(device))

            # Attention rollout: A_roll = prod_l( (A_l + I) / 2 )
            seq_len = 1 + grid * grid
            rollout = np.eye(seq_len, dtype=np.float32)
            for i in range(n_blocks):
                if i not in captured:
                    continue
                a = captured[i][0].numpy()  # (seq, seq)
                a = (a + np.eye(seq_len)) / 2.0
                a /= a.sum(axis=-1, keepdims=True) + 1e-8
                rollout = a @ rollout

            # CLS token row → patch attention (drop CLS→CLS at index 0)
            cls_attn = rollout[0, 1:]  # (196,)
            cam = cls_attn.reshape(grid, grid)
            cam = _resize_cam(cam, 32)
            cams.append(cam)

    for h in hooks:
        h.remove()

    return np.stack(cams)


def _resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray((cam / (cam.max() + 1e-8) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def extract(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vit_n_train = args.vit_n_train if args.vit_n_train is not None else args.n_train
    vit_epochs = args.vit_epochs if args.vit_epochs is not None else args.epochs

    train_loader, test_loader, cam_loader = _get_loaders(
        args.batch_size, args.n_train, args.n_test
    )

    raw_images_batch = next(iter(cam_loader))
    raw_images = raw_images_batch[0].numpy().transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
    cam_labels_raw = raw_images_batch[1].tolist()

    out_dir = Path("data/vision")

    # ── ResNet-18 ────────────────────────────────────────────────────────────
    print("\n=== ResNet-18 ===")
    resnet = _build_resnet(device)
    print(f"Fine-tuning for {args.epochs} epochs...")
    _train(resnet, train_loader, args.epochs, device)

    print("Extracting activations...")
    cnn_acts, cnn_labels = _extract_resnet_activations(resnet, test_loader, device)

    print("Computing GradCAM...")
    norm_tf = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)
    cam_images_norm = torch.stack(
        [norm_tf(torch.from_numpy(img.transpose(2, 0, 1))) for img in raw_images]
    )
    cnn_cams = _gradcam_resnet(resnet, cam_images_norm, cam_labels_raw, device)

    save_vision_data(
        out_dir / "resnet18.h5", cnn_acts, cnn_labels, raw_images, cnn_cams
    )
    print(
        f"Saved {len(cnn_acts)} layers × {len(cnn_labels)} samples → {out_dir}/resnet18.h5"
    )

    del resnet, cnn_acts, train_loader, test_loader
    torch.cuda.empty_cache()

    # ── ViT-B/16 ─────────────────────────────────────────────────────────────
    print("\n=== ViT-B/16 ===")
    vit_batch = max(8, args.batch_size // 4)  # 224×224 images need more VRAM
    vit_train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    vit_test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    vit_train_ds = datasets.CIFAR10("data/cifar10", train=True, transform=vit_train_tf)
    vit_test_ds = datasets.CIFAR10("data/cifar10", train=False, transform=vit_test_tf)
    vit_train_loader = DataLoader(
        Subset(vit_train_ds, range(vit_n_train)),
        batch_size=vit_batch,
        shuffle=True,
        num_workers=4,
    )
    vit_test_loader = DataLoader(
        Subset(vit_test_ds, range(args.n_test)),
        batch_size=vit_batch,
        shuffle=False,
        num_workers=4,
    )

    vit = _build_vit(device)
    print(f"Fine-tuning for {vit_epochs} epochs on {vit_n_train} samples...")
    _train(vit, vit_train_loader, vit_epochs, device)

    print("Extracting activations...")
    vit_acts, vit_labels = _extract_vit_activations(vit, vit_test_loader, device)

    print("Computing attention maps...")
    vit_cams = _attention_cam_vit(vit, cam_images_norm, device)

    save_vision_data(out_dir / "vit_b16.h5", vit_acts, vit_labels, raw_images, vit_cams)
    print(
        f"Saved {len(vit_acts)} layers × {len(vit_labels)} samples → {out_dir}/vit_b16.h5"
    )

    print("\nDone. Launch the dashboard with:")
    print("  python app/vision_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CNN+ViT on CIFAR-10 and extract embeddings."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Fine-tuning epochs (ResNet)"
    )
    parser.add_argument(
        "--n-train", type=int, default=10000, help="Training samples (ResNet)"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Test samples for activation extraction",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--vit-epochs",
        type=int,
        default=None,
        help="ViT fine-tuning epochs (defaults to --epochs)",
    )
    parser.add_argument(
        "--vit-n-train",
        type=int,
        default=None,
        help="ViT training samples (defaults to --n-train)",
    )
    extract(parser.parse_args())
