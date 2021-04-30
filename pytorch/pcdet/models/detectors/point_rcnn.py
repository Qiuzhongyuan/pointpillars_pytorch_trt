from .detector3d_template import Detector3DTemplate
import torch

class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.load_pretrained('/home/qiuzhongyuan/deeplearning/OpenPCDet/output/cfgs/kitti_models/pointrcnn_iou/default/ckpt/checkpoint_epoch_80.pth')
    def load_pretrained(self, path):
        state = torch.load(path)['model_state']
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k not in state:
                print(k, 'not in state')
                state[k] = self_state_dict[k].cpu()
            else:
                state[k] = state[k].cpu()

        self.load_state_dict(state)

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
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
