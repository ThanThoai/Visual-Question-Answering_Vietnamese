import torch.nn as nn


class BaseAdapter(nn.Module):
    
    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        self.dataset_init(__C)
        
    
    def dataset_init(self, __C):
        raise NotImplementedError()
    
    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_dict(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)
        
        return self.dataset_forward(feat_dict)
    
    def dataset_forward(self, feat_dict):
        raise NotImplementedError()
        