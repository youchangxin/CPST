import torch
import torch.nn as nn
import numpy as np


def get_wav(in_channels):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    net = nn.Conv2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2,
             padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2,
             padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2,
             padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2,
             padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class ContentEncoder(nn.Module):
    def __init__(self, encoder, disable_wavelet):
        super(ContentEncoder, self).__init__()
        self.disable_wavelet = disable_wavelet

        self.vgg = nn.Sequential(*encoder[:31])

        self.enc1 = nn.Sequential(*encoder[:7])
        self.pool1 = WavePool(64)
        self.enc2 = nn.Sequential(*encoder[8:14])
        self.pool2 = WavePool(128)
        self.enc3 = nn.Sequential(*encoder[15:27])
        self.pool3 = WavePool(256)
        self.enc4 = nn.Sequential(*encoder[28:31])

    def forward(self, input, skips):
        if self.disable_wavelet:
            return self.vgg(input)

        out = self.enc1(input)
        LL, LH, HL, HH = self.pool1(out)
        skips['pool1'] = [LH, HL, HH]

        out = self.enc2(LL)
        LL, LH, HL, HH = self.pool2(out)
        skips['pool2'] = [LH, HL, HH]

        out = self.enc3(LL)
        LL, LH, HL, HH = self.pool3(out)
        skips['pool3'] = [LH, HL, HH]

        out = self.enc4(LL)
        return out


class StyleEncoder(nn.Module):
    def __init__(self, encoder):
        super(StyleEncoder, self).__init__()
        self.styleEnc = nn.Sequential(*encoder[:31])  # input -> relu4_1 512

    def forward(self, input):
        results = self.styleEnc(input)
        return results


class Encoder(nn.Module):
    def __init__(self, disable_wavelet):
        super(Encoder, self).__init__()

        self.conEnc = ContentEncoder(self.backbone(disable_wavelet), disable_wavelet)
        self.styEnc = StyleEncoder(self.backbone())

    def backbone(self, disable_wavelet=True):
        backbone = [
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True) if disable_wavelet else WavePool(64),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True) if disable_wavelet else WavePool(128),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True) if disable_wavelet else WavePool(256),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True) if disable_wavelet else WavePool(512),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        ]
        return backbone

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content, style, skips, encoded_only=False):
        style_feats = self.styEnc(style)
        content_feats = self.conEnc(content, skips)
        if encoded_only:
            return content_feats, style_feats
        else:
            adain_feat = self.adain(content_feats, style_feats)
            return adain_feat


class Decoder(nn.Module):
    def __init__(self, disable_wavelet):
        super(Decoder, self).__init__()
        self.disable_wavelet = disable_wavelet
        decoder = [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),  # 256
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),  # 128
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        ]
        if self.disable_wavelet:
            self.whole_decoder = nn.Sequential(*decoder)
        self.dec1 = nn.Sequential(*decoder[:3])
        self.dec2 = nn.Sequential(*decoder[3:16])
        self.dec3 = nn.Sequential(*decoder[16:26])
        self.dec4 = nn.Sequential(*decoder[26:])

        self.att1 = Attention(256)
        self.att2 = Attention(128)
        self.att3 = Attention(64)

    def forward(self, adain_feat, skips):
        if self.disable_wavelet:
            return self.whole_decoder(adain_feat)
        out = self.dec1(adain_feat)

        #hf = torch.cat(skips["pool3"], dim=1)
        hf = torch.sum(torch.stack(skips["pool3"]), dim=0)
        out = torch.add(out, self.att1(hf))
        out = self.dec2(out)

        #hf = torch.cat(skips["pool2"], dim=1)
        hf = torch.sum(torch.stack(skips["pool2"]), dim=0)
        out = torch.add(out, self.att2(hf))
        out = self.dec3(out)

        #hf = torch.cat(skips["pool1"], dim=1)
        hf = torch.sum(torch.stack(skips["pool1"]), dim=0)
        out = torch.add(out, self.att3(hf))
        out = self.dec4(out)

        return out


class Attention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Attention, self).__init__()
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3)
        self.in1 = nn.InstanceNorm2d(channel)
        self.relu = nn.ReLU()
        #self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        #self.in2 = nn.InstanceNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(channel, channel // reduction, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(channel // reduction, channel, kernel_size=3),
        )
        self.sf = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.pad(input)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        #x = self.conv2(x)
        #x = self.in2(x)
        residual = x
        b, c, _, _ = residual.size()

        # channel attention
        y = self.squeeze(x)
        y = self.avg_pool(y).view(b, c, 1, 1)
        A = self.sf(y)
        return residual + x * A


class WaveletAttention(nn.Module):
    def __init__(self, in_planes):
        super(WaveletAttention, self).__init__()
        self.q = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.k = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.v = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, input):
        Q = self.q(input)
        K = self.k(input)
        V = self.v(input)

        b, c, h, w = Q.size()
        Q = Q.view(b, -1, w * h).permute(0, 2, 1)
        K = K.view(b, -1, w * h)
        S = torch.bmm(Q, K)
        S = self.sm(S)

        V = V.view(b, -1, w * h)
        O = torch.bmm(V, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        #output += V
        return O


