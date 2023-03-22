import torch
import torch.nn as nn
import numpy as np

from .networks import calc_mean_std, mean_variance_norm, get_key, adain_transform


def get_wav(in_channels, pool=True):
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

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

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
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=True)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class ContentEncoder(nn.Module):
    def __init__(self, enc_layers, disable_wavelet):
        super(ContentEncoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers[:44])
        self.enc_layers.load_state_dict(torch.load("vgg_normalised.pth"), strict=False)
        for param in self.enc_layers.parameters():
            param.requires_grad = False
        self.disable_wavelet = disable_wavelet

    def forward(self, x):
        relu_feats = []
        high_feats = {}
        max_feats = []
        pool_idxs = [7, 14, 27, 40]
        relu_x_idxs = [3, 10, 17, 30, 43]
        pool_idx = 1
        for idx, layer in enumerate(self.enc_layers):
            if idx in pool_idxs:
                if self.disable_wavelet:
                    x = layer(x)
                    max_feats.append(x)
                else:
                    LL, LH, HL, HH = layer(x)
                    x = LL
                    high_feats["pool"+str(pool_idx)] = (LH, HL, HH)
                    pool_idx += 1
            elif idx in relu_x_idxs:
                x = layer(x)
                relu_feats.append(x)

            else:
                x = layer(x)

        if self.disable_wavelet:
            return relu_feats, max_feats
        else:
            return relu_feats, high_feats


class StyleEncoder(nn.Module):
    def __init__(self, enc_layers):
        super(StyleEncoder, self).__init__()
        self.enc_layers = nn.Sequential(*enc_layers[:44])
        self.enc_layers.load_state_dict(torch.load("style_vgg.pth"), strict=False)

        for layer in self.enc_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        relu_feats = []
        pool_feats = []
        pool_idxs = [7, 14, 27, 40]
        relu_x_idxs = [3, 10, 17, 30, 43]
        for idx, layer in enumerate(self.enc_layers):
            if idx in pool_idxs:
                x = layer(x)
                pool_feats.append(x)
            elif idx in relu_x_idxs:
                x = layer(x)
                relu_feats.append(x)
            else:
                x = layer(x)

        return relu_feats, pool_feats


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

    def forward(self, content, style):
        style_feats, s_pool_feats = self.styEnc(style)
        content_feats, high_feats = self.conEnc(content)
        return style_feats, content_feats, high_feats, s_pool_feats


class AdaAttN(nn.Module):
    def __init__(self, in_planes,  key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, 1)
        self.g = nn.Conv2d(key_planes, key_planes, 1)
        self.h = nn.Conv2d(in_planes, in_planes, 1)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, c_x, s_x, c_1x, s_1x):
        F = self.f(c_1x)
        G = self.g(s_1x)
        H = self.h(s_x)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * c_x + mean


class Transformer(nn.Module):
    def __init__(self, shallow_layer=False):
        super().__init__()
        self.net_adaattn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if shallow_layer else 256)
        self.net_adaattn_4 = AdaAttN(in_planes=512, key_planes=512 + 256 + 128 + 64 if shallow_layer else 512)
        self.net_adaattn_5 = AdaAttN(in_planes=512, key_planes=512 + 512 + 256 + 128 + 64 if shallow_layer else 512)

        self.shallow_layer = shallow_layer
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(512, 512, (3, 3))

    def forward(self, c_feats, s_feats, h_feats, s_pool_feats):
        c_3 = mean_variance_norm(c_feats[2])
        c_4 = mean_variance_norm(c_feats[3])
        c_5 = mean_variance_norm(c_feats[4])

        pool_adain_feats = {}
        if isinstance(h_feats, list):
            for i in range(1, 4):
                pool_adain_feats["pool"+str(i)] = adain_transform(h_feats[i-1], s_pool_feats[i-1])
        else:
            for i in range(1, 4):
                pool_adain_feats["pool"+str(i)] = [adain_transform(h_feats["pool"+str(i)][0], s_pool_feats[i-1]),
                                                   adain_transform(h_feats["pool"+str(i)][1], s_pool_feats[i-1]),
                                                   adain_transform(h_feats["pool"+str(i)][2], s_pool_feats[i-1])]

        adain_feat_3 = self.net_adaattn_3(c_3, s_feats[2],
                                          get_key(c_feats, 2, self.shallow_layer),
                                          get_key(s_feats, 2, self.shallow_layer),)

        adain_feat_4 = self.net_adaattn_4(c_4, s_feats[3],
                                          get_key(c_feats, 3, self.shallow_layer),
                                          get_key(s_feats, 3, self.shallow_layer),)

        adain_feat_5 = self.net_adaattn_5(c_5, s_feats[4],
                                          get_key(c_feats, 4, self.shallow_layer),
                                          get_key(s_feats, 4, self.shallow_layer),)
        cs_feat = self.merge_conv(self.merge_conv_pad(adain_feat_4 + self.upsample5_1(adain_feat_5)))

        return cs_feat, adain_feat_3, pool_adain_feats


class Decoder(nn.Module):
    def __init__(self, skip_connection_3=False, disable_wavelet=False):
        super(Decoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.pool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3_4 = nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.pool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.pool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

        self.wavelet_attn_1 = WaveletAttention(64)
        self.wavelet_attn_2 = WaveletAttention(128)
        self.wavelet_attn_3 = WaveletAttention(256)

        self.skip_connection_3 = skip_connection_3
        self.disable_wavelet = disable_wavelet

    def decode(self, x, skips, level, adain_feat_3):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            if self.disable_wavelet:
                h_att = self.wavelet_attn_3(skips['pool3'])
                out = self.pool3(out + h_att)
            else:
                lh, hl, hh = skips['pool3']
                h_att = self.wavelet_attn_3(lh + hl + hh)
                out = self.pool3(out + h_att)
            if self.skip_connection_3:
                out = torch.cat((out, adain_feat_3), dim=1)
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))

        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            if self.disable_wavelet:
                h_att = self.wavelet_attn_2(skips['pool2'])
                out = self.pool2(out+h_att)
            else:
                lh, hl, hh = skips['pool2']
                h_att = self.wavelet_attn_2(lh + hl + hh)
                out = self.pool2(out + h_att)
            return self.relu(self.conv2_2(self.pad(out)))

        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            if self.disable_wavelet:
                h_att = self.wavelet_attn_1(skips['pool1'])
                out = self.pool1(out+h_att)
            else:
                lh, hl, hh = skips['pool1']
                h_att = self.wavelet_attn_1(lh + hl + hh)
                out = self.pool1(out + h_att)
            return self.relu(self.conv1_2(self.pad(out)))

        else:
            return self.conv1_1(self.pad(x))

    def forward(self, x, skips, adain_feat_3=None):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level, adain_feat_3)
        return x


class WaveletAttention(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(WaveletAttention, self).__init__()

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3)
        self.in1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size=3),
        )
        self.sf = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.pad(input)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        residual = x
        b, c, _, _ = residual.size()

        # channel attention
        y = self.squeeze(x)
        y = self.avg_pool(y).view(b, c, 1, 1)
        A = self.sf(y)
        return residual + x * A
