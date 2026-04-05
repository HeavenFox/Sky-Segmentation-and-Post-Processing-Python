"""U²-Net Small (U2NetP) architecture for sky segmentation.

Reimplemented from NCNN param file (skysegsmall_sim-opt-fp16.param).
No BatchNorm — the NCNN model has BN fused into Conv weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class RSU(nn.Module):
    """Residual U-block with pooling-based downsampling.

    Used for RSU-7, RSU-6, RSU-5, RSU-4 (L=7,6,5,4).
    Encoder: L-1 conv levels with MaxPool between them + 1 dilated bottom.
    Decoder: L-1 conv levels with bilinear upsample + concat.
    """

    def __init__(self, L, in_ch, mid_ch, out_ch):
        super().__init__()
        self.input_conv = ConvReLU(in_ch, out_ch)

        # Encoder: level 0 (no pool), levels 1..L-2 (with pool), bottom (dilated)
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvReLU(out_ch, mid_ch))
        for _ in range(1, L - 1):
            self.encoders.append(ConvReLU(mid_ch, mid_ch))
        self.bottom = ConvReLU(mid_ch, mid_ch, padding=2, dilation=2)

        # Decoder: L-1 levels (concat with each encoder level back to level 0)
        # First L-2 output mid_ch, last outputs out_ch
        self.decoders = nn.ModuleList()
        for i in range(L - 1):
            if i < L - 2:
                self.decoders.append(ConvReLU(2 * mid_ch, mid_ch))
            else:
                self.decoders.append(ConvReLU(2 * mid_ch, out_ch))

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x0 = self.input_conv(x)

        # Encoder
        enc = []
        h = self.encoders[0](x0)
        enc.append(h)
        for i in range(1, len(self.encoders)):
            h = self.pool(h)
            h = self.encoders[i](h)
            enc.append(h)

        # Bottom (same spatial size as last encoder level)
        h = self.bottom(h)

        # Decoder
        for i, dec_conv in enumerate(self.decoders):
            enc_feat = enc[len(enc) - 1 - i]
            if i > 0:
                h = F.interpolate(h, size=enc_feat.shape[2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, enc_feat], dim=1)
            h = dec_conv(h)

        return h + x0


class RSU4F(nn.Module):
    """Residual U-block with dilation only (no pooling).

    4 encoder levels with dilation 1, 2, 4, 8.
    3 decoder levels with dilation 4, 2, 1.
    """

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.input_conv = ConvReLU(in_ch, out_ch)
        self.enc0 = ConvReLU(out_ch, mid_ch, padding=1, dilation=1)
        self.enc1 = ConvReLU(mid_ch, mid_ch, padding=2, dilation=2)
        self.enc2 = ConvReLU(mid_ch, mid_ch, padding=4, dilation=4)
        self.enc3 = ConvReLU(mid_ch, mid_ch, padding=8, dilation=8)
        self.dec0 = ConvReLU(2 * mid_ch, mid_ch, padding=4, dilation=4)
        self.dec1 = ConvReLU(2 * mid_ch, mid_ch, padding=2, dilation=2)
        self.dec2 = ConvReLU(2 * mid_ch, out_ch, padding=1, dilation=1)

    def forward(self, x):
        x0 = self.input_conv(x)
        e0 = self.enc0(x0)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d0 = self.dec0(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d0, e1], dim=1))
        d2 = self.dec2(torch.cat([d1, e0], dim=1))
        return d2 + x0


class U2NetP(nn.Module):
    """U²-Net Small (U2NetP) for sky segmentation.

    Architecture from skysegsmall_sim-opt-fp16.param:
    - Encoder: RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F, RSU-4F
    - Decoder: RSU-4F, RSU-4, RSU-5, RSU-6, RSU-7
    - 6 side outputs (3x3 conv) fused by 1x1 conv + sigmoid
    """

    def __init__(self, in_ch=3, mid_ch=16, out_ch=64):
        super().__init__()
        # Encoder
        self.en1 = RSU(7, in_ch, mid_ch, out_ch)
        self.en2 = RSU(6, out_ch, mid_ch, out_ch)
        self.en3 = RSU(5, out_ch, mid_ch, out_ch)
        self.en4 = RSU(4, out_ch, mid_ch, out_ch)
        self.en5 = RSU4F(out_ch, mid_ch, out_ch)
        self.en6 = RSU4F(out_ch, mid_ch, out_ch)

        # Decoder
        self.de5 = RSU4F(2 * out_ch, mid_ch, out_ch)
        self.de4 = RSU(4, 2 * out_ch, mid_ch, out_ch)
        self.de3 = RSU(5, 2 * out_ch, mid_ch, out_ch)
        self.de2 = RSU(6, 2 * out_ch, mid_ch, out_ch)
        self.de1 = RSU(7, 2 * out_ch, mid_ch, out_ch)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Side output convolutions (3x3, no activation)
        self.side1 = nn.Conv2d(out_ch, 1, 3, padding=1)
        self.side2 = nn.Conv2d(out_ch, 1, 3, padding=1)
        self.side3 = nn.Conv2d(out_ch, 1, 3, padding=1)
        self.side4 = nn.Conv2d(out_ch, 1, 3, padding=1)
        self.side5 = nn.Conv2d(out_ch, 1, 3, padding=1)
        self.side6 = nn.Conv2d(out_ch, 1, 3, padding=1)

        # Fusion: 6 side outputs → 1 channel + sigmoid
        self.fuse = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]

        # Encoder
        x1 = self.en1(x)
        x2 = self.en2(self.pool(x1))
        x3 = self.en3(self.pool(x2))
        x4 = self.en4(self.pool(x3))
        x5 = self.en5(self.pool(x4))
        x6 = self.en6(self.pool(x5))

        # Decoder
        up6 = F.interpolate(x6, size=x5.shape[2:], mode='bilinear', align_corners=False)
        d5 = self.de5(torch.cat([up6, x5], dim=1))

        up5 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.de4(torch.cat([up5, x4], dim=1))

        up4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.de3(torch.cat([up4, x3], dim=1))

        up3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.de2(torch.cat([up3, x2], dim=1))

        up2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.de1(torch.cat([up2, x1], dim=1))

        # Side outputs
        s1 = self.side1(d1)
        s2 = F.interpolate(self.side2(d2), size=(H, W), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(d3), size=(H, W), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(d4), size=(H, W), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.side5(d5), size=(H, W), mode='bilinear', align_corners=False)
        s6 = F.interpolate(self.side6(x6), size=(H, W), mode='bilinear', align_corners=False)

        # Fusion
        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))
        fused = torch.sigmoid(fused)

        return fused, *[torch.sigmoid(s) for s in [s1, s2, s3, s4, s5, s6]]
