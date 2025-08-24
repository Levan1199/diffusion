import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
# your modules
from models.diffusion.networks import VPPrecond
from models.diffusion.loss import VPLoss
from models.diffusion.utils import tokens_to_map, estimate_latent_stats, LatentNormalizer, pad_to_multiple
from models.swin_ae.swin_ae import SwinAutoencoder

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="outputs/diffusion")
    p.add_argument("--load-ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume-optimizer", type=bool, default=False, help="Resume optimizer")
    p.add_argument("--cond-class", type=int, default=0, help="Number of conditional class labels")
    p.add_argument("--num-workers", type=int, default=4, help="Number of cpus")
    p.add_argument("--swin-path", type=str, default="", help="load pretrained encoder")
    return p.parse_args()

def build_model(device, cond_class, shape, channel):
    net = VPPrecond(
        img_resolution=shape,
        img_channels=channel,        # CIFAR10
        label_dim=cond_class,           # 10 conditional CIFAR10
        model_type='SongUNet',
        model_channels=64,
        channel_mult=[1, 2, 2],
        attn_resolutions=[8],
    ).to(device)
    return net




def load_checkpoint(ckpt_path, model, optimizer, device, resume_opt):
    ckpt = torch.load(ckpt_path, map_location=device)
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
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] -> [-1,1]
    ])

    

    swin_ae = SwinAutoencoder()
    ckpt_swin_ae =  torch.load(args.swin_path)
    swin_ae.load_state_dict(ckpt_swin_ae["model_state_dict"])
    swin_ae.to(args.device).eval()


    train_dataset = CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    
    with torch.no_grad():
        x0, _ = next(iter(train_loader))
        z0 = tokens_to_map(swin_ae.encoder(x0.to(args.device)))
    latent_mean, latent_std = estimate_latent_stats(swin_ae.encoder, train_loader, args.device, max_batches=200)
    norm = LatentNormalizer(latent_mean, latent_std).to(args.device)
    
    with torch.no_grad():
        z0n = norm(z0)                                            # [B,192,7,7]
        z0p, pad_spec0 = pad_to_multiple(z0n, m=8)                # [B,192,8,8]

    
    C_lat = z0p.shape[1]  # 192
    H_lat = z0p.shape[2]  # 8
    print(f"z0 {z0.shape},  z0 pad {z0p.shape}")
    

    model = build_model(args.device, args.cond_class, H_lat, C_lat)
    loss_fn = VPLoss(beta_d=19.9, beta_min=0.1, epsilon_t=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.load_ckpt:
        print(f"[resume] {args.load_ckpt}")
        load_checkpoint(args.load_ckpt, model, optimizer, args.device, args.resume_optimizer)

    print(f"start training, cuda: {torch.cuda.is_available()}, using device {args.device}, num_workers: {args.num_workers}, classes {args.cond_class}")
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), "VRAM:", torch.cuda.get_device_properties(i).total_memory/1024**3, "GB")
    
    epoch = 0
    for epoch in range(args.epochs):
        model.train()        
        running_loss = 0.0
        # pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)

        # for imgs, class_label in tqdm(pbar):
        for imgs, class_label in train_loader:
            imgs = imgs.to(args.device)

            if args.cond_class > 0:
                y = F.one_hot(class_label, num_classes=args.cond_class).float().to(args.device)
            else:
                y = None

            with torch.no_grad():
                latent_img = swin_ae.encoder(imgs)
            latent_img = tokens_to_map(latent_img)
            latent_img = norm(latent_img)       
            latent_img, pad_spec = pad_to_multiple(latent_img, m=8) 
            
            loss = loss_fn(net=model, images=latent_img, labels=y, augment_pipe=None).mean()

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
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg,
            },os.path.join(args.output_dir, f"ckpt_03_{epoch}.pt"))

if __name__ == "__main__":
    main()
