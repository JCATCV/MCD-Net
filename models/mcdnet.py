
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_norm import spectral_norm as _spectral_norm
from models.modules.tfocal_transformer import SoftSplit, SoftComp
from models.modules.depth_aware_transformer import DepthAwareTransformerBlock
from models.modules.spectral_norm import spectral_norm as _spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out

class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)

def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        init_He(self)
        self.activation = activation

    def forward(self, input):
        # O = act(Feature) * sig(Gating)
        feature = self.input_conv(input)
        if self.activation:
            feature = self.activation(feature)
        gating = torch.sigmoid(self.gating_conv(input))
        return feature * gating


class SpatialReductionTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2, dilation=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
 
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm0 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(proj_drop)
 
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, dilation=dilation)
            self.sr_2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio + 1, dilation=dilation)
            self.sr_3 = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 2, stride=sr_ratio + 2, dilation=dilation)
            self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4 * dim, dim)
        )
    def forward(self, x, H, W):
        BT, N, D = x.shape  #N=h*w
        shortcut = x
        x = self.norm0(x)
        q = self.q(x).reshape(BT, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
    
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(BT, D, 15, 27)
            x1 = self.sr_1(x_).reshape(BT, D, -1).permute(0, 2, 1) 
            x2 = self.sr_2(x_).reshape(BT, D, -1).permute(0, 2, 1)
            x3 = self.sr_3(x_).reshape(BT, D, -1).permute(0, 2, 1)
            x = torch.cat((x1, x2, x3), 1)
            x = self.norm1(x)
            kv = self.kv(x).reshape(1, -1, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        x = (attn @ v).transpose(1, 2).reshape(BT, N, D)
        x = self.proj(x)
        x = self.dropout(x) + shortcut

        shortcut = x
        x = shortcut + self.mlp(x)
        return x


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 8
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        self.ss = SoftSplit(channel // 2, hidden, kernel_size, stride, padding, t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size, stride, padding)

        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)

        depths = 8
        num_heads = [4] * depths
        window_size = [(5, 9)] * depths
        focal_windows = [(5, 9)] * depths
        focal_levels = [2] * depths
        pool_method = "fc"
        for i in range(depths):
            blocks.append(DepthAwareTransformerBlock(
                dim=hidden, num_heads=num_heads[i], window_size=window_size[i],
                focal_level=focal_levels[i], focal_window=focal_windows[i],
                n_vecs=n_vecs, t2t_params=t2t_params, pool_method=pool_method
            ))
        self.transformer = nn.Sequential(*blocks)
        self.encoder = Encoder()

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )
        


        self.fusenet1, self.fusenet2 = Fuse3D(), Fuse3D()

        self.depth_encoder = nn.Sequential(
            GatedConv2d(3, 64, 5, stride=2, padding=2, activation=nn.LeakyReLU(negative_slope=0.2)),
            GatedConv2d(64, 64, 3, 1, 1, activation=nn.LeakyReLU(negative_slope=0.2)),
            GatedConv2d(64, 128, 3, 2, 1, activation=nn.LeakyReLU(negative_slope=0.2)),
            GatedConv2d(128, 128, 3, 1, 1, activation=nn.LeakyReLU(negative_slope=0.2)),
        )


        self.spatial_filter = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(16, 128, kernel_size=7, padding=3)
        )

        self.feature2token = nn.Conv2d(128, 128, 4, 4, 0)
        
        self.temporal_filter = SpatialReductionTemporalAttention(dim=128)

        self.token2feature = nn.ConvTranspose2d(128, 128, 4, 4, 0)

        self.geometric_encoding = nn.Conv2d(128, 40, 1, 1, 0)

        self.depth_decoder = nn.Sequential(
            deconv(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

        self.maxpool = nn.AdaptiveMaxPool2d(1)

        if init_weights:
            self.init_weights()

        
        

    def forward(self, frames, depth, mask_rgb, mask_dep, in_edge):
        # ---------------------------------------------data preparation--------------------------------------------------

        b, t, c, h, w = frames.size()
        frames, depth, mask_rgb, mask_dep = frames.reshape(b * t, c, h, w), depth.reshape(b * t, 1, h, w), mask_rgb.reshape(b * t, 1, h, w), mask_dep.reshape(b * t, 1, h, w)
        masked_frames, masked_depth = frames * (1-mask_rgb), depth * (1-mask_dep)

        # ------------------------------------------------encoder layer--------------------------------------------------
        
        enc_feat = self.encoder(masked_frames)
        enc_depth = self.depth_encoder(torch.cat([masked_depth, in_edge, mask_dep], dim=1))
        shortcut = enc_depth
        _, c, h, w = enc_feat.size()

        # ---------------------------------------------depth inpainting--------------------------------------------------

        shortcut = enc_depth
        enc_depth = shortcut * torch.sigmoid(self.spatial_filter(enc_depth))
        tokens = self.feature2token(enc_depth).flatten(2).transpose(-1,-2)
        tokens = self.temporal_filter(tokens, h, w)
        tokens = tokens.transpose(-1,-2).reshape(-1, 128, h//4, w//4)
        enc_depth = self.token2feature(tokens)

        # ----------------------------------------------fusing depth to rgb feature-------------------------------------
        mask_rgb = F.interpolate(mask_rgb, scale_factor=1 / 4, mode='bilinear', align_corners=True, recompute_scale_factor=True)

        min_val, max_val = torch.min(enc_depth.flatten(2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1),  torch.max(enc_depth.flatten(2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        guidance_map = (enc_depth - min_val)/(max_val - min_val) * 2. - 1.
        enc_feat = self.fusenet1(enc_feat, guidance_map)
        # ----------------------------------------------color inpainting------------------------------------------------

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b)
        input = (trans_feat, self.geometric_encoding(guidance_map))
        trans_feat = self.transformer(input)[0]
        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat.reshape(b * t, -1, h, w)

        # ----------------------------------------------fusing depth to depth feature-------------------------------------

        guidance_map = (enc_feat + 1.)/2.
        enc_depth = self.fusenet2(enc_depth, guidance_map)
        # ---------------------------------------------------decoder layer-----------------------------------------------
        
        frame, depth = self.decoder(enc_feat), self.depth_decoder(enc_depth)
        # -----------------------------------------------------output layer----------------------------------------------
        frame, depth = torch.tanh(frame), torch.sigmoid(depth)
        return frame, depth


# #############################################################################
# ############################# Coarse Depth Inpaintorr  #############################
# #############################################################################

class Fuse3D(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, hidden//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden, hidden//2, 1, 1, 0)

        self.fusion_3d = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.rate_estimator = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden, 3, 1, 1)
        )
    def forward(self, main_feat, sub_feat):
        shortcut = main_feat
        main_feat = self.conv1(main_feat)
        sub_feat = self.conv2(sub_feat)
        fused_feat = self.fusion_3d(torch.cat((main_feat, sub_feat), 1).unsqueeze(0).transpose(1,2))
        fused_feat = fused_feat.squeeze(0).transpose(0,1)
        activation = torch.sigmoid(self.rate_estimator(fused_feat))
        output = shortcut + fused_feat * activation
        return output
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) 

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
