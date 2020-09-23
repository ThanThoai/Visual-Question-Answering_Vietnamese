__all__ = ['make_mask', 'feat_filter']

import torch


def make_mask(feature):
    return (torch.sum(torch.abs(feature), dim = -1) == 0).unsqueeze(1).unsqueeze(2)



def feat_filter(dataset, frcn_feat, bbox_feat):
    feat_dict = {}
    feat_dict['FastRCNN_FEAT'] = frcn_feat
    feat_dict['BBOX_FEAT']     = bbox_feat
    return feat_dict

