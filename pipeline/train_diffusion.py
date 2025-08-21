import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# your modules
from models.diffusion.networks import VPPrecond
from models.diffusion.loss import VPLoss

def parse_args():
    p = argparse.ArgumentParser(description="Train EDM (VP) on MNIST")
    p.add_argument("--save-dir", type=str, required=True,
                   help="Directory to save checkpoints & logs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint .pt to resume from (optional)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def build_model(device):
    net = VPPrecond(
        img_resolution=32,
        img_channels=1,        # MNIST is grayscale
        label_dim=0,           # unconditional
        model_type='SongUNet',
        model_channels=64,
        channel_mult=[1, 2, 2],
        attn_resolutions=[8],
    ).to(device)
    return net

def build_data(batch, num_workers):
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])
    ds = datasets.MNIST(root="mnist_data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    return loader

def save_checkpoint(save_dir, epoch, net, opt, loss_val):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": net.state_dict(),
        "optimizer_state": opt.state_dict(),
        "loss": float(loss_val),
    }, ckpt_path)
    # also write/overwrite a latest pointer for convenience
    return ckpt_path

def load_checkpoint(path, net, opt, device):
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    if opt is not None and "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1  # continue after saved epoch
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    net = build_model(device)
    loss_fn = VPLoss(beta_d=19.9, beta_min=0.1, epsilon_t=1e-5)
    opt = optim.AdamW(net.parameters(), lr=args.lr)

    loader = build_data(args.batch, args.num_workers)

    # resume if requested
    start_epoch = 1
    if args.resume is not None and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, net, opt, device)

    net.train()
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        last_loss = None
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)

            loss = loss_fn(net=net, images=x, labels=None, augment_pipe=None).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            last_loss = float(loss.item())
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        print(f"epoch {epoch} | loss {last_loss:.4f}")
        ckpt_path = save_checkpoint(args.save_dir, epoch, net, opt, last_loss)
        print(f"saved: {ckpt_path}")

if __name__ == "__main__":
    main()
