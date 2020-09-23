from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import sys

sys.path.append('../')

from core import BaseAdapter
from core.utils import *
from core.ops import *


class BANAdapter(BaseAdapter):
    
    def __init__(self, __C):
        
        super(BANAdapter, self).__init__(__C)
        self.__C = __C
        
    def dataset_init(self, __C):
        
        self.frcn_linear = nn.Linear(__C.FEAT_SIZE['FRCM_FEAT_SIZE'][1], __C.HIDDEN_SIZE)
        
    
    def dataset_forward(self, feat_dict):
        
        frcn_feat = feat_dict['FastRCNN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        img_feat_mask = make_mask(frcn_feat)
        
        img_feat = self.frcn_linear(frcn_feat)
        return frcn_feat, img_feat_mask
    
class WeightNormMLP(nn.Module):
    
    """
        Weight Normal MLP
    """
    
    def __init__(self, dims, activation = 'ReLu', dropout_r = 0.):
        super(WeightNormMLP, self).__init__()
        
        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim = None))
            
            if activation != '':
                layers.append(getattr(nn, activation)())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)
    

class BilinearConnect(nn.Module):
    
    def __init__(self, __C, attention = False):
        super(BilinearConnect, self).__init__()
        self.__C = __C
        self.v_net = WeightNormMLP(dims = [self.__C.IMG_FEAT_SIZE, self.__C.BA_HIDDEN_SIZE], dropout_r = self.__C.DROPOUT_R)
        pass 
    

    