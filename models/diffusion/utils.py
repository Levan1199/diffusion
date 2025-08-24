import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def tokens_to_map(z):
    # Accept [B,N,C] -> [B,C,H,W]; pass through if already [B,C,H,W]
    if z.dim() == 3:
        B, N, C = z.shape
        H = W = int(math.isqrt(N))
        assert H * W == N, f"Tokens ({N}) not square."
        z = z.transpose(1, 2).contiguous().view(B, C, H, W)
    return z



@torch.no_grad()
def estimate_latent_stats(encoder, loader, device, max_batches=200):
    # running mean / var (Welford)
    n = 0
    mean = None
    M2 = None
    for i, (imgs, _) in enumerate(loader):
        if i >= max_batches: break
        imgs = imgs.to(device)
        z = encoder(imgs)             # [B,N,C] or [B,C,H,W]
        z = tokens_to_map(z)          # [B,C,H,W]
        B, C, H, W = z.shape
        x = z.permute(1,0,2,3).reshape(C, -1)  # [C, B*H*W]

        if mean is None:
            mean = x.mean(dim=1)
            M2   = ((x - mean[:,None])**2).sum(dim=1)
            n    = x.shape[1]
        else:
            new_n = x.shape[1]
            delta = x.mean(dim=1) - mean
            total_n = n + new_n
            mean = mean + delta * (new_n / total_n)
            M2 = M2 + ((x - mean[:,None])**2).sum(dim=1) + (delta**2) * (n*new_n/total_n)
            n = total_n

    var = M2 / max(n-1, 1)
    std = torch.sqrt(var + 1e-6)
    return mean.to(device), std.to(device)

class LatentNormalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std",  std.view(1, -1, 1, 1))
    def forward(self, z):
        return (z - self.mean) / self.std

def pad_to_multiple(x, m=8):
    B, C, H, W = x.shape
    Hp = ((H + m - 1) // m) * m
    Wp = ((W + m - 1) // m) * m
    pad = (0, Wp - W, 0, Hp - H)  # (left, right, top, bottom) -> here pad right/bottom
    return F.pad(x, pad), pad  # return pad spec if you want to unpad later


def map_to_tokens(x):
    # [B,C,H,W] -> [B,N,C]
    B, C, H, W = x.shape
    return x.view(B, C, H*W).transpose(1, 2).contiguous()

def unpad_hw(x, pad):  # pad = (left, right, top, bottom)
    l, r, t, b = pad
    return x[..., t:x.shape[-2]-b, l:x.shape[-1]-r]

@torch.no_grad()
def latent_to_decoder_tokens(
    z_lat,                      # [B,C_lat,8,8] from diffusion
    latent_mean, latent_std,    # from your estimator (shape [C] each)
    pad_spec=(0,1,0,1),         # you padded right/bottom by 1 to go 7->8
    rev_proj: nn.Module = None  # e.g., nn.Conv2d(C_lat, 192, 1) or None
):
    # 1) unpad 8x8 -> 7x7
    z = unpad_hw(z_lat, pad_spec)            # [B,C_lat,7,7]

    # 2) restore channel dim expected by decoder
    if rev_proj is not None:                 # if you reduced channels during training
        z = rev_proj(z)                      # -> [B,192,7,7]

    # 3) de-normalize back to encoder's distribution
    mean = latent_mean.view(1, -1, 1, 1).to(z.device)
    std  = latent_std.view(1, -1, 1, 1).to(z.device)
    z = z * std + mean                       # [B,192,7,7]

    # 4) reshape to tokens for decoder
    tokens = map_to_tokens(z)                # [B,49,192]
    return tokens