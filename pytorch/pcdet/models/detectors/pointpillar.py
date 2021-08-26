from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def pruned(self):
        import math
        module_pruned_conv = [(10, 20), (20, 39), \
                              (39, 39), (39, 39), (39, 52), (52, 32), \
                              (32, 26), (26, 13), (13, 13), (13, 39), (39, 52), (52, 39), \
                              (39, 26), (26, 52), (52, 26), (26, 26), (26, 26), (26, 26), \
                              (39, 7), (32, 20), (39, 10), (26, 10), (40, 116), \
                              40, 40, 40, 123, 123, 123]
        iter = 0
        bn_layer = -1
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                orign_shape = module.weight.shape
                device = module.weight.device
                out_channels = orign_shape[0]
                kernel = orign_shape[2]
                cfg = module_pruned_conv[iter]
                if isinstance(cfg, (list, tuple)):
                    in_channels, out_channels = cfg
                else:
                    in_channels = cfg
                module.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel, kernel)).to(device)
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                module.in_channels = in_channels
                module.out_channels = out_channels
                bn_layer = out_channels
                iter += 1
            elif isinstance(module, nn.BatchNorm2d):
                module.num_features = bn_layer
                module.num_features = bn_layer
                module.weight = nn.Parameter(torch.ones(bn_layer))
                module.bias = nn.Parameter(torch.zeros(bn_layer))
                module.running_mean = module.running_mean.data.clone()[:bn_layer]
                module.running_var = module.running_var.data.clone()[:bn_layer]

            elif isinstance(module, nn.ConvTranspose2d):
                orign_shape = module.weight.shape
                device = module.weight.device
                kernel = orign_shape[2]
                cfg = module_pruned_conv[iter]
                in_channels, out_channels = cfg

                kernel_size = module.weight.size(2)
                module.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels, kernel_size, kernel_size)).to(device)
                module.in_channels = in_channels
                module.out_channels = out_channels
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                bn_layer = out_channels
                iter += 1

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict