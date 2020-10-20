import numpy as np
import glob 
import json
import re
import torch

from core.datasets import Dataset


class DataLoader(Dataset):
    def __init__(self, __C):
        super(DataLoader, self).__init__()
        self.__C = __C
        
        ques_dict_preread = {
            'train' : json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'rb')),
            'val'   : json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'rb')),
            'test'  : json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'rb'))
        }
        
        frcn_feat_path_list = glob.glob(__C.FEATS_PATH[__.DATASET]['default-frcn'] + '/*.npz')
        grid_feat_path_list = glob.glob(__C.FEATS_PATH[__.DATASET]['default-grid'] + '/*.npz')