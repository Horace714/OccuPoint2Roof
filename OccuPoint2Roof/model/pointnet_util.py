import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import pc_util

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int, wd: float = 1.0, wf: float = 0.0) -> torch.Tensor:

        xyz = xyz.contiguous()

        B, N, C = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pc_util.furthest_point_sampling_wrapper(B, C, N, npoint, wd, wf, xyz, temp, output)

        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()

furthest_point_sample = FurthestPointSampling.apply

class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

        features = features.contiguous()
        idx = idx.contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pc_util.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.save_for_backwards = (idx, features)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        B, npoint = idx.size()
        _, C, N = features.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pc_util.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None

gather_operation = GatherOperation.apply

class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        unknown = unknown.contiguous()
        known = known.contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pc_util.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return ()

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        features = features.contiguous()
        idx = idx.contiguous()
        weight = weight.contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.save_for_backward(idx, weight, features)
        output = torch.cuda.FloatTensor(B, c, n)

        pc_util.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        idx, weight, features = ctx.saved_tensors
        m = features.size(2)
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pc_util.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)

three_interpolate = ThreeInterpolate.apply

class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

        features = features.contiguous()
        idx = idx.contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pc_util.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.save_for_backward(idx, features)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        idx, features = ctx.saved_tensors
        N = features.size(2)

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pc_util.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, torch.zeros_like(idx)

grouping_operation = GroupingOperation.apply

class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:

        new_xyz = new_xyz.contiguous()
        xyz = xyz.contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pc_util.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out):
        return ()

ball_query = BallQuery.apply

class BallCenterQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, point: torch.Tensor, key_point: torch.Tensor) -> torch.Tensor:

        point = point.contiguous()
        key_point = key_point.contiguous()

        B, N, _ = point.size()
        npoint = key_point.size(1)
        idx = torch.cuda.IntTensor(B, N).zero_() - 1

        pc_util.ball_center_query_wrapper(B, N, npoint, radius, point, key_point, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out):
        return ()

ball_center_query = BallCenterQuery.apply

import numpy as np

class KNNQuery(Function):

    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        new_xyz = new_xyz.contiguous()
        xyz = xyz.contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        dist2 = torch.cuda.FloatTensor(np.ones([B, npoint, nsample]) * 1e4)
        idx = torch.cuda.IntTensor(B, npoint, nsample)

        pc_util.knn_query_wrapper(B, N, npoint, nsample, new_xyz, xyz, dist2, idx)
        ctx.mark_non_differentiable(dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None

knn_query = KNNQuery.apply

