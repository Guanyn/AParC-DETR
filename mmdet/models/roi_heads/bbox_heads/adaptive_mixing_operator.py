import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
from timm.models.layers import trunc_normal_, DropPath


def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)


_dump_i = 0


# this requires a customized MMCV to include
# the flops updated in `__user_flops_handle__`
class SRShadowForFlops(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, **kwargs):
        super(SRShadowForFlops, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

    def forward(self, x, query):
        pass

    @staticmethod
    def __user_flops_handle__(module, input, output):
        B, num_query, num_group, num_point, num_channel = input[0].shape

        eff_in_dim = module.in_dim // num_group
        eff_out_dim = module.out_dim // num_group
        in_points = module.in_points
        out_points = module.out_points

        step1 = B * num_query * num_group * in_points * eff_in_dim * eff_out_dim
        step2 = B * num_query * num_group * eff_out_dim * in_points * out_points
        module.__flops__ += int(step1 + step2)
        pass


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, sampling_rate=None):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points // sampling_rate
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points

        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.n_groups * self.total_parameters),
        )

        self.out_proj = nn.Linear(
            self.eff_out_dim * self.out_points * self.n_groups, self.query_dim, bias=True
        )

        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.GELU()

        # virtual modules for FLOPs calculation
        local_dict = locals()
        local_dict.pop('self')
        self.shadow = SRShadowForFlops(**local_dict)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator[-1].weight)

    def forward(self, x, query):
        # Calculate FLOPs
        self.shadow(x, query)
        B, N, g, P, C = x.size()
        # batch, num_query, group, point, channel
        G = self.n_groups
        assert g == G
        # assert C*g == self.in_dim

        # query: B, N, C
        # x: B, N, G, Px, Cx

        global _dump_i

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B * N, G, -1)

        out = x.reshape(B * N, G, P, C)

        M, S = params.split(
            [self.m_parameters, self.s_parameters], 2)

        '''you can choose one implementation below'''
        if False:
            out = out.reshape(
                B * N * G, P, C
            )

            M = M.reshape(
                B * N * G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(
                B * N * G, self.out_points, self.in_points)

            '''adaptive channel mixing'''
            out = torch.bmm(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = AdaptMLP(out)
            out = self.act2(out)

            '''adaptive spatial mixing'''
            out = torch.bmm(S, out)  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = ParC_ViT(out)
            #out = CycleMLP(out)
            out = self.act(out)
        else:
            M = M.reshape(
                B * N, G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(
                B * N, G, self.out_points, self.in_points)

            '''adaptive channel mixing'''

            out = torch.matmul(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = AdaptMLP(out)
            out = self.act2(out)

            '''adaptive spatial mixing'''
            out = torch.matmul(S, out)  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = ParC_ViT(out)
            #out = ParC_operator(out)
            out = self.act2(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        out = query + out
        out = self.act(out)
        return out


class CycleFC(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        """
        这里的kernel_size实际使用的时候时3x1或者1x3
        """
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # 被偏移调整的1x1卷积的权重，由于后面使用torchvision提供的可变形卷积的函数，所以权重需要自己构造
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))
        # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # 要注意，这里是在注册一个buffer，是一个常量，不可学习，但是可以保存到模型权重中。
        self.register_buffer('offset', self.gen_offset())

    def gen_offset(self):

        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                # 这里计算了一个相对偏移位置。
                # deform_conv2d使用的以对应输出位置为中心的偏移坐标索引方式
                offset[0, 2 * i + 1, 0, 0] = (
                        (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
                )
            else:
                offset[0, 2 * i + 0, 0, 0] = (
                        (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                )
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input,
                                self.offset.expand(B, -1, H, W),
                                self.weight,
                                self.bias,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1, 5), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (5, 1), 1, 0)

        #self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        #self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AdaptMLP(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim=64, dropout=0.0, s=0.03):
        super().__init__()
        self.original_mlp = Mlp # original MLP block
    # down --> non linear --> up
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s # scaling factor
        # initialization
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        # freeze original MLP
        for _, p in self.original_mlp.named_parameters():
            p.requires_grad = False
    def forward(self, x):
        down = self.down_proj(x)
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        output = self.original_mlp(x) + up * self.scale
        return output

class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size, use_pe=True):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        if use_pe:
            if self.type == 'H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type == 'W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]),
                                                                                              dim=3)
        x = self.gcc_conv(x_cat)

        return x

class ParC_ViT(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, global_kernel_size=14, use_pe=True):
        self.gcc_H = ParC_operator(dim // 2, 'H', global_kernel_size, use_pe)
        self.gcc_W = ParC_operator(dim // 2, 'W', global_kernel_size, use_pe)
        self.Pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                              requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        #Frame4 x_1 = self.gcc_H(self.gcc_W(x_1))
        #Frame4 x_2 = self.gcc_H(self.gcc_W(x_2))
        #Frame3 x_1 = self.gcc_W(self.gcc_H(x_1))
        #Frame3 x_2 = self.gcc_W(self.gcc_H(x_2))
        #Frame2 x_1 = self.gcc_H(x_1)
        #Frame2 x_2 = self.gcc_W(x_2)
        #Frmae5
        x_1 = self.gcc_W(self.gcc_H(x_1))
        #Frmae5
        x_2 = self.gcc_H(self.gcc_W(x_2))
        x = input + self.drop_path(torch.cat((x_1, x_2), dim=1))
        return x
