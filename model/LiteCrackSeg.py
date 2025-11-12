import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from transformers import MobileViTModel
from timm.models.layers import DropPath, trunc_normal_

class DWConv(nn.Module):
    def __init__(self, dim=768,group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim//group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class eca_layer(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(eca_layer, self).__init__()
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


import warnings

warnings.filterwarnings("ignore")


class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        super(DSConv, self).__init__()

        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size


        self.dsc_conv_x = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        B, C, H, W = f.shape
        cur_device = f.device
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        
        dsc = DSC(f.shape, self.kernel_size, self.extend_scope, self.morph,
                  cur_device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        

        if self.morph == 0: 

            deformed_feature = rearrange(deformed_feature, 'b c (w k) h -> b c (k h) w', k=self.kernel_size)
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:

            deformed_feature = rearrange(deformed_feature, 'b c w (h k) -> b c h (k w)', k=self.kernel_size)
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope

        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]


    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:

            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:

            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([], device=self.device).int()
        max_y = self.width - 1
        max_x = self.height - 1


        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1


        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1


        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)


        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1


        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature




class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1 = nn.GroupNorm(max(1, hidden_features//4), hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(max(1, hidden_features // 4), hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3 = nn.GroupNorm(max(1, out_features//4), out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.gn3(x)
        x = self.drop(x)
        return x

class LocalSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=7):
        super(LocalSABlock, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.GroupNorm(k*heads//4,k*heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
            nn.GroupNorm(k*u//4,k*u)
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.GroupNorm(self.vv*u//4,self.vv*u)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)  
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) 
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)  
        content = torch.einsum('bkum,bvum->bkv', (softmax, values))
        content = torch.einsum('bhkn,bkv->bhvn', (queries, content))
        values = values.view(n_batch, self.uu, -1, w, h)
        context = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
        context = context.view(n_batch, self.kk, self.vv, w * h)
        context = torch.einsum('bhkn,bkvn->bhvn', (queries, context))

        out = content + context
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, mlp_ratio=2., drop=0.1,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        self.attn = LocalSABlock(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = max(planes,in_planes) // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(hidden_planes //4,
                                hidden_planes)  
        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        self.bn2 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)  
        self.conv2.append(nn.GELU()) 
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes // 4, planes)  
        self.GELU=nn.GELU()
        self.shortcut = nn.Sequential()
        if in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.GroupNorm(planes//4,planes)
            )
    def forward(self, x):
        out = self.GELU(self.bn1(self.conv1(x)))  
        out = self.conv2(out)
        out = self.GELU(self.bn3(self.conv3(out))) 
        out += self.shortcut(x)
        return out


class Trans_DB(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Bottleneck(in_, out)
        self.activation=torch.nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class CABlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=True),
            nn.GroupNorm(max(1, output_channels // 4), output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=True),
            nn.GroupNorm(max(1, output_channels // 4), output_channels),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, inputs):
        if not inputs:
            raise ValueError("CABlock got an empty input list")

        target_size = inputs[0].shape[2:]        
        aligned = []
        for feat in inputs:
            if feat.shape[2:] != target_size:     
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False)
            aligned.append(feat)

        fused = self.act(sum(aligned))
        fused = self.W_1(fused)
        return self.psi(fused)                


crack_channels = [48, 64, 96, 128, 160]
fuse_channels  = 8        
attn_channels   = 8 


class Fuse(nn.Module):
    def __init__(self, in_ch):                     
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, fuse_channels, 1, 1, 0, bias=False)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(fuse_channels, 1, 1)

    def forward(self, low, up, size, att):
        if up.shape[2:] != low.shape[2:]:
            up = F.interpolate(up, size=low.shape[2:], mode='bilinear', align_corners=False)
        x  = torch.cat([low, up], 1)       
        x  = self.pw(self.act(self.dw(x)))  
        x  = self.out(att * x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Up1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.nn = Trans_DB(crack_channels[1], crack_channels[0])  # 96->64

    def forward(self, x):
        x = self.up(x)
        y = self.nn(x)
        return y, y


class Up2(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.nn = Trans_DB(crack_channels[2], crack_channels[1])  # 128->96

    def forward(self, x):
        x = self.up(x); y = self.nn(x); return y, y


class Up3(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.nn = Trans_DB(crack_channels[3], crack_channels[2])  # 160->128

    def forward(self, x):
        x = self.up(x); y = self.nn(x); return y, y


class Up4(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.nn = Trans_DB(crack_channels[4], crack_channels[3])  # 192->160

    def forward(self, x):
        x = self.up(x); y = self.nn(x); return y, y


class Up5(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.nn = Trans_DB(crack_channels[4], crack_channels[4])  # 192->192

    def forward(self, x):
        x = self.up(x); y = self.nn(x); return y, y



def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        _, _, h, w = x.shape
        pad_bottom = (self.ph - h % self.ph) % self.ph
        pad_right = (self.pw - w % self.pw) % self.pw

        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='constant', value=0)  

        h_padded, w_padded = x.shape[2], x.shape[3]

        x = rearrange(x, "b d (h ph) (w pw) -> b (ph pw) (h w) d", ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, "b (ph pw) (h w) d -> b d (h ph) (w pw)", h=h_padded // self.ph, w=w_padded // self.pw, ph=self.ph, pw=self.pw)

        x = x[:, :, :h, :w]  

        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MAMViT(nn.Module):
    """
    MV2 ↓2  ➜  dual-morph DSConv  ➜  MobileViT block
    """
    def __init__(self, dims, channels, depths,
                 kernel_size, patch_size, expansion):
        super().__init__()

        self.conv_down = MV2Block(channels[0], channels[1],
                                  stride=2, expansion=expansion)

 
        ch = channels[1]              
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dsc_x = DSConv(ch, ch, kernel_size=5,
                            extend_scope=1.0, morph=0, if_offset=True,
                            device=device)
        self.dsc_y = DSConv(ch, ch, kernel_size=5,
                            extend_scope=1.0, morph=1, if_offset=True,
                            device=device)
        self.merge = nn.Conv2d(2 * ch, ch, 1, bias=False)


        self.attn = MobileViTBlock(dims[0], depths[0], channels[2],
                                   kernel_size, patch_size, int(dims[0] * 4))

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv_down(x)      

        x = torch.cat([self.dsc_x(x), self.dsc_y(x)], dim=1)
        x = self.merge(x)
    
        x = self.attn(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MViTxxsEncoderPretrained(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = MobileViTModel.from_pretrained("/home/Kaleb/LiteCrackSeg/model/mobilevit-xx-small", local_files_only=True).base_model # local path to mobilevit-xxs location

    def forward(self, x):
        raw_input = x.clone()
        hidden_states = self.encoder(x, output_hidden_states=True).hidden_states
        return {"raw_input": raw_input, "hidden_states": hidden_states}
    

class ChannelAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        hidden_ch = in_ch * 2  
        self.expand = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.act = nn.SiLU()
        self.project = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.expand(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.project(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class LiteCrackSeg(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = MViTxxsEncoderPretrained()

     
        
        bottleneck_params = {
            'dims': [144],
            'depths': [3],
            'channels': [80, 144, 144],
            'kernel_size': 3,
            'patch_size': (2, 2),
            'expansion': 4
        }
        self.bottleneck = MAMViT(**bottleneck_params)
        self.btl_adapter = nn.Conv2d(144, crack_channels[4], 1) # 144 -> 160

        mob_channels = [16, 24, 48, 64, 80]
        self.adapters = nn.ModuleList([
            ChannelAdapter(mob_ch, crack_ch) for mob_ch, crack_ch in zip(mob_channels, crack_channels)
        ])

        self.eca_blocks = nn.ModuleList([
            eca_layer(in_channels=ch) for ch in crack_channels 
        ])


 
        self.up1 = Up1()
        self.up2 = Up2()
        self.up3 = Up3()
        self.up4 = Up4()
        self.up5 = Up5()

        self.fuse5 = Fuse(crack_channels[4] * 2)   
        self.fuse4 = Fuse(crack_channels[3] * 2)  
        self.fuse3 = Fuse(crack_channels[2] * 2)  
        self.fuse2 = Fuse(crack_channels[1] * 2)   
        self.fuse1 = Fuse(crack_channels[0] * 2)  



        self.final = Conv1X1(5, 1)

        self.CABlock_1 = CABlock(crack_channels[0], attn_channels)
        self.CABlock_2 = CABlock(crack_channels[1], attn_channels)
        self.CABlock_3 = CABlock(crack_channels[2], attn_channels)
        self.CABlock_4 = CABlock(crack_channels[3], attn_channels)
        self.CABlock_5 = CABlock(crack_channels[4], attn_channels)

    def forward(self, inputs):
        enc_dict = self.encoder(inputs)
        hidden_states = enc_dict['hidden_states']
        if len(hidden_states) != 5:
            raise ValueError(f"Expected 5 hidden states, got {len(hidden_states)}")

        features = hidden_states[-5:]
        adapted_features = [self.adapters[i](feat) for i, feat in enumerate(features)]

        refined_features = [self.eca_blocks[i](feat) for i, feat in enumerate(adapted_features)]

        scale1_1 = scale1_2 = refined_features[0]
        scale2_1 = scale2_2 = refined_features[1]
        scale3_1 = scale3_2 = scale3_3 = refined_features[2]
        scale4_1 = scale4_2 = scale4_3 = refined_features[3]
        scale5_1 = scale5_2 = scale5_3 = refined_features[4]

        deepest = self.bottleneck(hidden_states[-1])
        deepest = self.btl_adapter(deepest)

        scale5_4, up5 = self.up5(deepest)
        scale4_4, up4 = self.up4(up5)
        scale3_4, up3 = self.up3(up4)
        scale2_3, up2 = self.up2(up3)
        scale1_3, up1 = self.up1(up2)


        att1 = self.CABlock_1([scale1_1, scale1_3])
        att2 = self.CABlock_2([scale2_1, scale2_3])
        att3 = self.CABlock_3([scale3_1, scale3_2, scale3_4])
        att4 = self.CABlock_4([scale4_1, scale4_2, scale4_4])
        att5 = self.CABlock_5([scale5_1, scale5_2, scale5_4])


        input_size = [inputs.shape[2], inputs.shape[3]]
        
        f5 = self.fuse5(scale5_3, up5, input_size, att5)
        f4 = self.fuse4(scale4_3, up4, input_size, att4)
        f3 = self.fuse3(scale3_3, up3, input_size, att3)
        f2 = self.fuse2(scale2_2, up2, input_size, att2)
        f1 = self.fuse1(scale1_2, up1, input_size, att1)


        output = self.final(torch.cat([f5, f4, f3, f2, f1], dim=1))

        return output, f5, f4, f3, f2, f1


from thop import profile, clever_format
import torch
if __name__ == "__main__":
    inp = torch.randn(1, 3, 512, 512)
    model = LiteCrackSeg()
    _ = model(inp)

    macs, params = profile(model, inputs=(inp,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    macs_value = float(macs.replace("G", "")) 
    flops_value = macs_value * 2              

    print(f"FLOPs: {flops_value:.3f} G")
    print(f"Parameters: {params}")

