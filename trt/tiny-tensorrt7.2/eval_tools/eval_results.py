import argparse
import pcdet
from pcdet.datasets import KittiDataset
import torch
import os
import numpy as np
import pickle
import sys
from kitti_object_eval_python import eval

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_infos', type=str, default='./val_infos', help='')
    parser.add_argument('--gts', type=str, default='eval_gt_annos.pkl', help='')
    parser.add_argument('--trt_outputs', type=str, default=None, required=True, help='')
    args = parser.parse_args()

    return args

def load_preds(path, car_head=True):
    data = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, 9)).float()
    num = (data[:, -1] > -0.5).sum()
    if num == 0:
        return torch.zeros((0, 7)), torch.zeros((0,)), torch.zeros((0,)).long()
        
    data = data[:num]
    if car_head:
        return data[:, :7], data[:, 7], (data[:, 8]+ 0.01).long() + 1
    else:
        return data[:, :7], data[:, 7], (data[:, 8]+ 0.01).long() + 2
    

def main():
    args = parse_config()
    
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    with open(args.gts, 'rb') as f:
        eval_gt_annos = pickle.load(f)
        
    frame_infos = os.listdir(args.frame_infos)
    eval_names = [name[:-4] for name in frame_infos]
    eval_names = sorted(eval_names)
    det_annos = []
    
    for name in eval_names:
        output_1 = os.path.join(args.trt_outputs, name + '_0.bin')
        box1, score1, type1 = load_preds(output_1, True)
        
        output_2 = os.path.join(args.trt_outputs, name + '_1.bin')
        box2, score2, type2 = load_preds(output_2, False)
        
        box = torch.cat([box1, box2], 0)
        score = torch.cat([score1, score2], 0)
        type = torch.cat([type1, type2], 0)
        pred_dicts = [{'pred_scores':score, 'pred_boxes':box, 'pred_labels':type}]
        with open(os.path.join(args.frame_infos, name + '.pkl'), 'rb') as f:
            batch_dict = pickle.load(f)
            
        annos = KittiDataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names)
        
        det_annos += annos

        # print(annos)
        # 1/0


    ap_result_str, ap_dict = eval.get_official_eval_result(eval_gt_annos, det_annos, class_names)
    print('**********************************************eval results******************************************************************')
    print(ap_result_str)
    print('**********************************************eval results******************************************************************')
if __name__ == '__main__':
    main()
