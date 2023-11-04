import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import collections


class BoxAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = ops.box_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        return output


def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    # assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list)
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        # print("=============", tensor.size(), '=============')
        # new_tensor = tensor.flatten(2).permute(0, 1)
        tensor_flatten.append(tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[0]
        tensor_shape[i, 1] = tensor.shape[1]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape


class BoxAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_level=4, kernel_size=2):
        super(BoxAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model should be divided by num_heads"
        self.im2col_step = 64
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_level = num_level
        self.head_dim = d_model // num_heads
        self.kernel_size = kernel_size
        self.num_point = kernel_size ** 2
        self.ref_size = 4

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_heads * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_heads * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_heads * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_heads * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices)
        #i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(b, l, self.num_heads, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        grid = center + self.kernel_indices * torch.relu(size)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    # def forward(
    #     #self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    #     self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    # ):
    #     b, l1 = query.shape[:2]
    #     l2 = value.shape[1]
    #
    #     value = self.value_proj(value)
    #     if v_mask is not None:
    #         value = value.masked_fill(v_mask[..., None], float(0))
    #     value = value.view(b, l2, self.num_heads, self.head_dim)
    #
    #     attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
    #     attn_weights = F.softmax(attn_weights.view(b, l1, self.num_heads, -1), dim=-1)
    #     attn_weights = attn_weights.view(
    #         b, l1, self.num_heads, self.num_level, self.kernel_size, self.kernel_size
    #     )
    #
    #     sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
    #     output = BoxAttnFunction.apply(
    #         value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
    #     )
    #     output = self.out_proj(output)
    #
    #     return output, attn_weights

    def forward(
            self, value, mask, pos
    ):
        assert pos is not None, "position encoding is required!"
        if mask is None or mask[0] is None:
            mask = None

        ref_windows = self._create_ref_windows(value, mask)
        v_valid_ratios = self._create_valid_ratios(value, mask)
        value, v_mask, v_shape = flatten_with_shape(value, mask)

        src_pos = []
        if pos[0] is not None:
            for pe in pos:
                b, c = pe.shape[:2]
                pe = pe.view(b, c, -1).transpose(1, 2)
                src_pos.append(pe)
            src_pos = torch.cat(src_pos, dim=1)
        else:
            assert self.adaptive_pe
            src_pos = None
        v_start_index = torch.cat(
            [v_shape.new_zeros(1), v_shape.prod(1).cumsum(0)[:-1]]
        )

        query = self.with_pos_embed(value, src_pos)

        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_heads, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_heads, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_heads, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _create_ref_windows(self, tensor_list, mask_list):
        ref_windows = []

        eps = 1e-6
        for i, tensor in enumerate(tensor_list):
            if mask_list is not None:
                not_mask = ~(mask_list[i])
                y_embed = not_mask.cumsum(1, dtype=tensor.dtype)
                x_embed = not_mask.cumsum(2, dtype=tensor.dtype)

                size_h = not_mask[:, :, 0].sum(dim=-1, dtype=tensor.dtype)
                size_w = not_mask[:, 0, :].sum(dim=-1, dtype=tensor.dtype)
            else:
                size_h, size_w = tensor.shape[-2:]
                y_embed = torch.arange(
                    1, size_h + 1, dtype=tensor.dtype, device=tensor.device
                )
                x_embed = torch.arange(
                    1, size_w + 1, dtype=tensor.dtype, device=tensor.device
                )
                y_embed, x_embed = torch.meshgrid(y_embed, x_embed)
                x_embed = x_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
                y_embed = y_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)

                size_h = torch.tensor(
                    [size_h] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )
                size_w = torch.tensor(
                    [size_w] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )

            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
            center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2)

            h_embed = self.ref_size / size_h
            w_embed = self.ref_size / size_w

            size = torch.stack([w_embed, h_embed], dim=-1)
            size = size.unsqueeze(1).expand_as(center)

            ref_box = torch.cat([center, size], dim=-1)
            ref_windows.append(ref_box)

        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _create_valid_ratios(self, src, masks):
        if masks is None:
            return None

        ratios = []
        for mask in masks:
            not_mask = ~mask
            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=src[0].dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=src[0].dtype)

            h, w = mask.shape[-2:]
            ratio_w = size_w / w
            ratio_h = size_h / h
            ratio = torch.stack([ratio_w, ratio_h], dim=-1)

            ratios.append(ratio)
        valid_ratios = (
            torch.stack(ratios, dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(-2)
        )

        return valid_ratios
