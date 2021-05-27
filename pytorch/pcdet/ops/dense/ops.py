import torch
from torch.autograd import Function
from . import dense_cuda

class DenseFunction(Function):
    @staticmethod
    def forward(ctx, features, coords, batch_size, spatialShape):
        features = features.contiguous()
        assert features.dim()== 2
        assert coords.dim() == 2 and  coords.size(1) == 4
        ctx.save_for_backward(coords)
        ctx.spatialShape = [batch_size] + list(spatialShape)
        output = dense_cuda.dense(features, coords, batch_size, spatialShape)
        return output
        
    @staticmethod
    def symbolic(g, features, coords, batch_size, spatialShape):
        return g.op('Dense', features, coords, batch_size_i=batch_size, spatialShape_i=spatialShape)
    
    @staticmethod
    def backward(ctx, grad_output):
        coords = ctx.saved_tensors[0]
        spatialShape = ctx.spatialShape[1:]
        input_bp = dense_cuda.dense_backward(grad_output, coords, spatialShape)
        return input_bp, None, None, None
        
Dense = DenseFunction.apply


    
    