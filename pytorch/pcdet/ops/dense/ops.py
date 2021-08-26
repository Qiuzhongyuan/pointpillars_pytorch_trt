import torch
from torch.autograd import Function
from . import dense_cuda


class DenseFunction(Function):
    @staticmethod
    def forward(ctx, features, coords, valid, spatialShape):
        assert features.dim() == 2
        assert coords.dim() == 2 and coords.size(1) == 3
        assert valid.dim() == 1
        batch_size = int(valid.size(0))
        features = features.contiguous()
        ctx.save_for_backward(coords, valid)
        ctx.spatialShape = list(spatialShape)
        output = dense_cuda.dense(features, coords, valid, batch_size, spatialShape)
        return output

    @staticmethod
    def symbolic(g, features, coords, valid, spatialShape):
        return g.op('Dense', features, coords, valid, spatialShape_i=spatialShape)

    @staticmethod
    def backward(ctx, grad_output):
        coords, valid = ctx.saved_tensors
        spatialShape = ctx.spatialShape
        input_bp = dense_cuda.dense_backward(grad_output, coords, valid, spatialShape)
        return input_bp, None, None, None

dense = DenseFunction.apply