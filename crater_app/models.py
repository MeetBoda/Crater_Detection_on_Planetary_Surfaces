import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tiling import split_into_tiles, stitch_tiles
import cv2
import streamlit as st

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   # Apple GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ===============================
# 🔥 YOUR MODEL ARCHITECTURE
# (UNCHANGED)
# ===============================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        half_ch = out_ch // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, half_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_ch),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(half_ch, half_ch, kernel_size=3, padding=1,
                      groups=half_ch, bias=False),
            nn.BatchNorm2d(half_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class ResidualGhostBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ghost1 = GhostModule(in_ch,  out_ch)
        self.ghost2 = GhostModule(out_ch, out_ch)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x   = self.ghost1(x)
        x   = self.ghost2(x)
        return F.relu(x + res, inplace=True)


class ECA_Layer(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size,
                                  padding=(k_size - 1) // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class DilatedConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DilatedResidualDenseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.d1 = DilatedConvBnRelu(out_ch, out_ch, dilation=1)
        self.d2 = DilatedConvBnRelu(out_ch, out_ch, dilation=2)
        self.d4 = DilatedConvBnRelu(out_ch, out_ch, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.compress(x)
        out = self.fuse(torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1))
        return F.relu(out + x, inplace=True)


class CBAMSkip(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.max_pool    = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        ca = torch.sigmoid(
            self.channel_mlp(self.avg_pool(x)) +
            self.channel_mlp(self.max_pool(x))
        ).view(B, C, 1, 1)
        x  = x * ca
        sa = torch.sigmoid(self.spatial_conv(
            torch.cat([x.mean(dim=1, keepdim=True),
                       x.max(dim=1,  keepdim=True).values], dim=1)))
        return x * sa


class WindowTransformerSkip(nn.Module):
    def __init__(self, channels: int, window_size: int = 8,
                 n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        while channels % n_heads != 0 and n_heads > 1: n_heads -= 1
        self.ws = window_size
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(channels)

    def _partition(self, x, ws):
        B, C, H, W = x.shape
        pH = (ws - H % ws) % ws; pW = (ws - W % ws) % ws
        if pH > 0 or pW > 0: x = F.pad(x, (0, pW, 0, pH))
        _, _, Hp, Wp = x.shape
        x = x.reshape(B, C, Hp//ws, ws, Wp//ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws*ws, C)
        return x, (Hp, Wp, pH, pW)

    def _unpartition(self, x, B, C, Hp, Wp, pH, pW, ws):
        nH, nW = Hp//ws, Wp//ws
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        if pH > 0 or pW > 0: x = x[:, :, :Hp-pH, :Wp-pW]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = min(self.ws, H, W)
        tokens, (Hp, Wp, pH, pW) = self._partition(x, ws)
        tokens  = self.out_norm(self.transformer(tokens))
        refined = self._unpartition(tokens, B, C, Hp, Wp, pH, pW, ws)
        return x + refined


class CrossScaleTransformerSkip(nn.Module):
    def __init__(self, channels_l3: int, channels_l4: int,
                 n_heads: int = 8, n_layers: int = 1, pool_size: int = 8):
        super().__init__()
        while channels_l3 % n_heads != 0 and n_heads > 1: n_heads -= 1
        self.pool_size = pool_size
        self.l4_proj   = nn.Conv2d(channels_l4, channels_l3, 1, bias=False)
        self.self_attn = WindowTransformerSkip(channels_l3, window_size=8,
                                               n_heads=n_heads, n_layers=n_layers)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels_l3, num_heads=n_heads, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(channels_l3)
        self.expand     = nn.Conv2d(channels_l3, channels_l3, 1, bias=False)

    def forward(self, x_l3: torch.Tensor, x_l4: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_l3.shape; ps = self.pool_size
        out  = self.self_attn(x_l3)
        l3_p = F.adaptive_avg_pool2d(out,                   (ps, ps))
        l4_p = F.adaptive_avg_pool2d(self.l4_proj(x_l4),   (ps, ps))
        q    = l3_p.flatten(2).permute(0, 2, 1)
        k    = l4_p.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.cross_attn(query=q, key=k, value=k)
        q = self.cross_norm(q + attn_out)
        cross_map = q.permute(0, 2, 1).reshape(B, C, ps, ps)
        cross_up  = F.interpolate(self.expand(cross_map), size=(H, W),
                                  mode='bilinear', align_corners=True)
        return out + cross_up


class CircularTransformerSkip(nn.Module):
    def __init__(self, channels: int, n_heads: int = 8, n_layers: int = 2,
                 window_size: int = 8):
        super().__init__()
        self.win_attn  = WindowTransformerSkip(channels, window_size=window_size,
                                               n_heads=n_heads, n_layers=n_layers)
        self.radial_mlp = nn.Sequential(
            nn.Conv2d(3, 32, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 1, bias=False), nn.Sigmoid(),
        )

    def _radial_coords(self, H, W, device):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        gr = (gx**2 + gy**2).sqrt() / (2**0.5)
        return torch.stack([gx, gy, gr], dim=0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out    = self.win_attn(x)
        coords = self._radial_coords(H, W, x.device)
        gate   = self.radial_mlp(coords.expand(B, -1, -1, -1))
        return x + out * gate


class GhostRDTUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.ghost_block = ResidualGhostBlock(out_ch, out_ch)
        self.eca = ECA_Layer(k_size=3)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.ghost_block(x)
        return self.eca(x)


class ChannelReducedMSFHead(nn.Module):
    def __init__(self, ch0, ch1, ch2, reduce_ch=32, out_channels=1):
        super().__init__()
        self.reduce_d2 = nn.Sequential(
            nn.Conv2d(ch1, reduce_ch, 1, bias=False),
            nn.BatchNorm2d(reduce_ch), nn.ReLU(inplace=True),
        )
        self.reduce_d3 = nn.Sequential(
            nn.Conv2d(ch2, reduce_ch, 1, bias=False),
            nn.BatchNorm2d(reduce_ch), nn.ReLU(inplace=True),
        )
        total = ch0 + reduce_ch + reduce_ch
        self.fuse = nn.Sequential(
            ConvBnRelu(total, ch0),
            ConvBnRelu(ch0, ch0 // 2),
            nn.Conv2d(ch0 // 2, out_channels, 1),
        )

    def forward(self, d1, d2, d3):
        H, W = d1.shape[2:]
        d2 = F.interpolate(self.reduce_d2(d2), size=(H, W), mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.reduce_d3(d3), size=(H, W), mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([d1, d2, d3], dim=1))


class AuxHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )

    def forward(self, x, target_size):
        return F.interpolate(self.head(x), size=target_size,
                             mode='bilinear', align_corners=True)


class GhostRDTUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(4)]

        self.stem = nn.Sequential(
            ConvBnRelu(in_channels, ch[0]),
            ConvBnRelu(ch[0], ch[0]),
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = DilatedResidualDenseBlock(ch[0], ch[0])
        self.enc2 = DilatedResidualDenseBlock(ch[0], ch[1])
        self.enc3 = DilatedResidualDenseBlock(ch[1] + ch[0], ch[2])
        self.enc4 = DilatedResidualDenseBlock(ch[2] + ch[1] + ch[0], ch[3])

        self.skip1 = CBAMSkip(ch[0])
        self.skip2 = CBAMSkip(ch[0])
        self.skip3 = CrossScaleTransformerSkip(ch[2], ch[3])
        self.skip4 = CircularTransformerSkip(ch[3])

        self.bottleneck = DilatedResidualDenseBlock(ch[3], ch[3])

        self.dec4 = GhostRDTUpBlock(ch[3], ch[3], ch[2])
        self.dec3 = GhostRDTUpBlock(ch[2], ch[2], ch[1])
        self.dec2 = GhostRDTUpBlock(ch[1], ch[0], ch[0])
        self.dec1 = GhostRDTUpBlock(ch[0], ch[0], ch[0])

        self.head = ChannelReducedMSFHead(ch[0], ch[0], ch[1])

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(self.pool(s))
        e2 = self.enc2(self.pool(e1))

        e1d = F.adaptive_avg_pool2d(e1, e2.shape[2:])
        e3 = self.enc3(self.pool(torch.cat([e2, e1d], dim=1)))

        e2d = F.adaptive_avg_pool2d(e2, e3.shape[2:])
        e1d2 = F.adaptive_avg_pool2d(e1, e3.shape[2:])
        e4 = self.enc4(self.pool(torch.cat([e3, e2d, e1d2], dim=1)))

        sk1 = self.skip1(s)
        sk2 = self.skip2(e1)
        sk3 = self.skip3(e3, e4)
        sk4 = self.skip4(e4)

        bn = self.bottleneck(e4)

        d4 = self.dec4(bn, sk4)
        d3 = self.dec3(d4, sk3)
        d2 = self.dec2(d3, sk2)
        d1 = self.dec1(d2, sk1)

        return self.head(d1, d2, d3)
    
class RDTUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.conv1 = ConvBnRelu(out_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x   = self.fuse(torch.cat([x, skip], dim=1))
        out = self.conv2(self.conv1(x))
        return F.relu(out + x, inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RDT-UNet++  — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RDTUNetPlusPlus(nn.Module):
    """
    RDT-UNet++ for Planetary Crater Segmentation.

    base_ch=32 gives channels [32, 64, 128, 256] — memory-safe on 15 GB.
    Increase to 48 or 64 only if you have headroom after sanity_check().

    Encoder : DilatedResidualDenseBlock (rates 1,2,4) at all 4 levels
    Skip L1 : CBAMSkip     (full res, 32ch)
    Skip L2 : CBAMSkip     (H/2, 64ch)
    Skip L3 : CrossScaleTransformerSkip (window self-attn + pooled cross-attn)
    Skip L4 : CircularTransformerSkip (window attn + radial geometry gate)
    Decoder : RDTUpBlock (residual) at all 4 levels
    Head    : ChannelReducedMSFHead (d1+d2_32ch+d3_32ch → prediction)
    Training: AuxHead at d2 and d3 (deep supervision, discarded at inference)

    Forward:
        train mode → (main_logit, aux_l3_logit, aux_l2_logit)
        eval mode  → main_logit
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_ch: int = 32, attn_heads: int = 8):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(4)]  # [32, 64, 128, 256]

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBnRelu(in_channels, ch[0]),
            ConvBnRelu(ch[0], ch[0]),
        )
        self.pool = nn.MaxPool2d(2, 2)

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = DilatedResidualDenseBlock(ch[0], ch[0])
        self.enc2 = DilatedResidualDenseBlock(ch[0], ch[1])
        self.enc3 = DilatedResidualDenseBlock(ch[1] + ch[0], ch[2])
        self.enc4 = DilatedResidualDenseBlock(ch[2] + ch[1] + ch[0], ch[3])

        # ── Skip attention ─────────────────────────────────────────────────
        self.skip1 = CBAMSkip(ch[0])
        self.skip2 = CBAMSkip(ch[1])
        self.skip3 = CrossScaleTransformerSkip(
            ch[2], ch[3], n_heads=min(attn_heads, ch[2] // 8),
            n_layers=1, pool_size=8
        )
        self.skip4 = CircularTransformerSkip(
            ch[3], n_heads=min(attn_heads, ch[3] // 8),
            n_layers=2, window_size=8
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec3 = RDTUpBlock(ch[3], ch[2], ch[2])
        self.dec2 = RDTUpBlock(ch[2], ch[1], ch[1])
        self.dec1 = RDTUpBlock(ch[1], ch[0], ch[0])

        # ── Head ──────────────────────────────────────────────────────────
        self.head = ChannelReducedMSFHead(
            ch[0], ch[1], ch[2],
            reduce_ch=32, out_channels=out_channels
        )

        # ── Auxiliary heads (training only) ───────────────────────────────
        self.aux_l3 = AuxHead(ch[2], out_channels)
        self.aux_l2 = AuxHead(ch[1], out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        H_in, W_in = x.shape[2:]

        # ── Encode ────────────────────────────────────────────────────────
        s  = self.stem(x)
        e1 = self.enc1(s)

        p1 = self.pool(e1)
        e2 = self.enc2(p1)

        p2, p1x2 = self.pool(e2), self.pool(p1)
        e3 = self.enc3(torch.cat([p2, p1x2], dim=1))

        p3, p2x2, p1x3 = self.pool(e3), self.pool(p2), self.pool(p1x2)
        e4 = self.enc4(torch.cat([p3, p2x2, p1x3], dim=1))

        # ── Skip connections ──────────────────────────────────────────────
        sk1 = self.skip1(e1)
        sk2 = self.skip2(e2)
        sk4 = self.skip4(e4)
        sk3 = self.skip3(e3, e4)   # cross-scale: L3 queries L4

        # ── Decode ────────────────────────────────────────────────────────
        d3 = self.dec3(sk4, sk3)
        d2 = self.dec2(d3,  sk2)
        d1 = self.dec1(d2,  sk1)

        # ── Main prediction ───────────────────────────────────────────────
        main = self.head(d1, d2, d3)

        if self.training:
            aux3 = self.aux_l3(d3, (H_in, W_in))
            aux2 = self.aux_l2(d2, (H_in, W_in))
            return main, aux3, aux2

        return main




# ===============================
# 🔥 MODEL LOADER
# ===============================
def load_model(model_name):
    if model_name == "Ghost-RDT-UNet++":
        model = GhostRDTUNetPlusPlus()
        filename = "ghost_rdt_unet++_aug_45epochs.pth"

    elif model_name == "RDT-UNet++":
        model = RDTUNetPlusPlus()
        filename = "rdt_unet++_aug_25epochs.pth"

    else:
        raise ValueError("Unknown model")

    path = os.path.join(os.path.dirname(__file__), "models", filename)

    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    model.to(DEVICE)
    model.eval()

    return model

def predict_large_image(model, image, tile_size=512):

    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tiles, positions, shape = split_into_tiles(image, tile_size)

    outputs = []

    # ✅ PROGRESS BAR
    progress = st.progress(0)
    status_text = st.empty()

    total = len(tiles)

    for i, tile in enumerate(tiles):
        status_text.text(f"Processing tile {i+1}/{total}")

        prob = predict(model, tile)
        outputs.append(prob)

        progress.progress((i + 1) / total)

    status_text.text("Stitching tiles...")

    full_prob_map = stitch_tiles(outputs, positions, shape)

    progress.empty()
    status_text.empty()

    return full_prob_map


# ===============================
# 🔥 PREDICT
# ===============================
@torch.no_grad()
def predict(model, image):
    import cv2

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
    img = img.to(DEVICE)

    output = model(img)

    return output.squeeze().cpu().numpy()