"""Run ViT-B/16 extraction only (ResNet-18 already saved)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from scripts.io import save_vision_data

device = torch.device("cuda")
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)
N_CAM, EPOCHS, N_TRAIN, N_TEST, BATCH = 40, 10, 10000, 1000, 16

vit_train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)
vit_test_tf = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
)
raw_tf = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.CIFAR10("data/cifar10", train=True, transform=vit_train_tf)
test_ds = datasets.CIFAR10("data/cifar10", train=False, transform=vit_test_tf)
raw_ds = datasets.CIFAR10("data/cifar10", train=False, transform=raw_tf)

train_loader = DataLoader(
    Subset(train_ds, range(N_TRAIN)), batch_size=BATCH, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    Subset(test_ds, range(N_TEST)), batch_size=BATCH, shuffle=False, num_workers=0
)
cam_loader = DataLoader(Subset(raw_ds, range(N_CAM)), batch_size=N_CAM, shuffle=False)

raw_batch = next(iter(cam_loader))
raw_images = raw_batch[0].numpy().transpose(0, 2, 3, 1)

model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
model.heads.head = nn.Linear(768, 10)
model = model.to(device)

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss()
for epoch in range(1, EPOCHS + 1):
    total = correct = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        crit(out, y).backward()
        opt.step()
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    sched.step()
    print(f"  epoch {epoch}/{EPOCHS}  train acc: {correct / total:.3f}")

hooks, buffers, emb_buf = [], {}, {}


def _blk_hook(i):
    def fn(_, __, out):
        buffers[i] = out[:, 0, :].detach().cpu()

    return fn


for i, blk in enumerate(model.encoder.layers):
    hooks.append(blk.register_forward_hook(_blk_hook(i)))
emb_h = model.conv_proj.register_forward_hook(
    lambda m, inp, out: emb_buf.update(
        {-1: out.flatten(2).transpose(1, 2)[:, 0, :].detach().cpu()}
    )
)

model.eval()
n_blocks = len(model.encoder.layers)
all_acts = {i: [] for i in range(-1, n_blocks)}
all_labels = []
with torch.no_grad():
    for X, y in test_loader:
        _ = model(X.to(device))
        all_acts[-1].append(emb_buf[-1].numpy())
        for i in range(n_blocks):
            all_acts[i].append(buffers[i].numpy())
        all_labels.extend(y.numpy())

for h in hooks:
    h.remove()
emb_h.remove()

activations = {
    j: np.concatenate(all_acts[k], axis=0)
    for j, k in enumerate(sorted(all_acts.keys()))
}
labels = np.array(all_labels, dtype=np.int8)
vit_cams = np.ones((N_CAM, 32, 32), dtype=np.float32)

save_vision_data(
    Path("data/vision/vit_b16.h5"), activations, labels, raw_images, vit_cams
)
print(
    f"Saved {len(activations)} layers x {len(labels)} samples -> data/vision/vit_b16.h5"
)
