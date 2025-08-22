# --- sampling.py (patch) ---
import argparse, os, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from models.diffusion.networks import VPPrecond

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def name_to_index(name: str) -> int:
    name = name.strip().lower()
    if name not in CIFAR10_CLASSES:
        raise ValueError(f"Unknown class name '{name}'. Choose from: {CIFAR10_CLASSES}")
    return CIFAR10_CLASSES.index(name)
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def build_model(device):
    net = VPPrecond(
        img_resolution=32,
        img_channels=3,        # MNIST grayscale
        label_dim=10,           # 0 = unconditional; set 10 if trained class-cond
        model_type='SongUNet',
        model_channels=64,
        channel_mult=[1, 2, 2],
        attn_resolutions=[8],
    ).to(device)
    return net
# ... keep edm_sampler() and build_model() as-is ...

def load_checkpoint_into(net, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    net.load_state_dict(state)
    net.eval().requires_grad_(False)
    return net
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="EDM Sampler")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="samples")
    parser.add_argument("--batch-size", type=int, default=16)
    # existing numeric label (kept for compatibility)
    parser.add_argument("--class-label", type=int, default=None,
                        help="Integer class id [0..9]; ignored if --class-name is given")
    # NEW: pass a class name or a comma-separated list
    parser.add_argument("--class-name", type=str, default=None,
                        help="e.g. 'frog' or 'airplane' or 'frog,cat,dog'")
    parser.add_argument("--num-steps", type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(device)  # label_dim=10 in your script
    load_checkpoint_into(net, args.ckpt_path, device)
    print(f"in main bla bla")
    # Make latents
    latents = torch.randn(args.batch_size, net.img_channels, net.img_resolution, net.img_resolution, device=device)

    # Build class_labels one-hot (or None for unconditional)
    if args.class_name is not None:
        names = [n for n in args.class_name.split(",") if n.strip()]
        idxs = [name_to_index(n) for n in names]
        # repeat names/ids to fill the batch
        idxs = (idxs * ((args.batch_size + len(idxs) - 1) // len(idxs)))[:args.batch_size]
        y = torch.tensor(idxs, device=device, dtype=torch.long)
        class_labels = F.one_hot(y, num_classes=10).float()
    elif args.class_label is not None:
        y = torch.full((args.batch_size,), int(args.class_label), device=device, dtype=torch.long)
        class_labels = F.one_hot(y, num_classes=10).float()
    else:
        class_labels = None  # unconditional

    imgs = edm_sampler(
        net=net,
        latents=latents,
        class_labels=class_labels,
        num_steps=args.num_steps,
        sigma_min=0.002, sigma_max=80, rho=7,
    )

    imgs_u8 = imgs.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for i, arr in enumerate(imgs_u8):
        print(f"saving {args.out_dir}")
        Image.fromarray(arr, mode="RGB").resize((256,256), Image.NEAREST).save(os.path.join(args.out_dir, f"sample_{i:03d}.png"))



if __name__ == "__main__":
    main()












#================================================
# import os
# import argparse
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from models.diffusion.networks import VPPrecond
# import numpy as np

# # --- paste edm_sampler() function here ---
# def edm_sampler(
#     net, latents, class_labels=None, randn_like=torch.randn_like,
#     num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
#     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
# ):
#     # Adjust noise levels based on what's supported by the network.
#     sigma_min = max(sigma_min, net.sigma_min)
#     sigma_max = min(sigma_max, net.sigma_max)

#     # Time step discretization.
#     step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
#     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

#     # Main sampling loop.
#     x_next = latents.to(torch.float64) * t_steps[0]
#     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
#         x_cur = x_next

#         # Increase noise temporarily.
#         gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
#         t_hat = net.round_sigma(t_cur + gamma * t_cur)
#         x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

#         # Euler step.
#         denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
#         d_cur = (x_hat - denoised) / t_hat
#         x_next = x_hat + (t_next - t_hat) * d_cur

#         # Apply 2nd order correction.
#         if i < num_steps - 1:
#             denoised = net(x_next, t_next, class_labels).to(torch.float64)
#             d_prime = (x_next - denoised) / t_next
#             x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

#     return x_next

# def build_model(device):
#     net = VPPrecond(
#         img_resolution=32,
#         img_channels=3,        # MNIST grayscale
#         label_dim=10,           # 0 = unconditional; set 10 if trained class-cond
#         model_type='SongUNet',
#         model_channels=64,
#         channel_mult=[1, 2, 2],
#         attn_resolutions=[8],
#     ).to(device)
#     return net

# def load_checkpoint_into(net, ckpt_path, device):
#     ckpt = torch.load(ckpt_path, map_location=device)
#     state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
#     net.load_state_dict(state)
#     net.eval().requires_grad_(False)
#     return net

# @torch.no_grad()
# def main():
#     parser = argparse.ArgumentParser(description="EDM Sampler")
#     parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the trained checkpoint")
#     parser.add_argument("--out-dir", type=str, default="samples", help="Where to save generated images")
#     parser.add_argument("--batch-size", type=int, default=16)
#     parser.add_argument("--class-label", type=int, default=0)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1) Load model + checkpoint
#     net = build_model(device)
#     load_checkpoint_into(net, args.ckpt_path, device)

#     # 2) Make latents
#     latents = torch.randn(args.batch_size, net.img_channels, net.img_resolution, net.img_resolution, device=device)

#     # repeat the same class for the whole batch
#     cls_lbl = torch.full((args.batch_size,), args.class_label,
#                         device=device, dtype=torch.long)
#     class_labels = F.one_hot(cls_lbl, num_classes=10).float()


#     # 3) Sample
#     imgs = edm_sampler(
#         net=net,
#         latents=latents,
#         class_labels=class_labels,
#         num_steps=18,
#         sigma_min=0.002,
#         sigma_max=80,
#         rho=7,
#     )
#     imgs_u8 = imgs.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)  # [B, 3, H, W]
#     imgs_u8 = imgs_u8.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]

#     for i, arr in enumerate(imgs_u8):
#         Image.fromarray(arr, mode="RGB").resize((256, 256), Image.BICUBIC).save(
#             os.path.join(args.out_dir, f"sample_{i:03d}.png"))


# if __name__ == "__main__":
#     main()