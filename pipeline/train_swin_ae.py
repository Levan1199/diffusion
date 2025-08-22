import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision
from models.swin_ae.swin_ae import SwinAutoencoder
from tqdm import tqdm
import os, glob, argparse, torch
# === Dataset ===

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="outputs/swin_ae")
    p.add_argument("--load-ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume-optimizer", type=bool, default=False, help="Resume optimizer")
    p.add_argument("--num-workers", type=int, default=4, help="Number of cpus")
    return p.parse_args()


def load_checkpoint(ckpt_path, model, optimizer, device, resume_opt):
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"old loss: {ckpt["loss"]}")
    if "model_state_dict" in ckpt:
        print(f"load model state dict")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt and resume_opt:
        print(f"load optimizer")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]) 

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    print(f"start training, cuda: {torch.cuda.is_available()} using device {args.device}, num_workers: {args.num_workers}")

    # === Model ===
    device = torch.device(args.device)
    model = SwinAutoencoder().to(device)

    # === Loss and Optimizer ===
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    if args.load_ckpt:
        print(f"[resume] {args.load_ckpt} ")
        load_checkpoint(args.load_ckpt, model, optimizer, device, args.resume_optimizer)

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), "VRAM:", torch.cuda.get_device_properties(i).total_memory/1024**3, "GB")



    # === Training Loop ===
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        # pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)

        # for imgs, _ in tqdm(pbar):
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running_loss / len(train_loader)
        print(f"epoch {epoch} | loss {avg:.4f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg,
            },os.path.join(args.output_dir, f"ckpt_03_{epoch}.pt"))


if __name__ == "__main__":
    main()