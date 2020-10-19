import os 
import numpy as np
import glob
import json
import torch
import random



def make_dir(dir, path_dir):
    if dir not in os.path.listdir(path_dir):
        os.mkdir(os.path.join(path_dir, dir))
        

def feat_filter(frcn_feat, grid_feat, bbox_feat):
    feat_dict = dict()
    feat_dict['FRCN_FEAT'] = frcn_feat
    feat_dict['GRID_FEAT'] = grid_feat
    frcn_feat['BBOX_FEAT'] = bbox_feat
    
    return feat_dict


class PATH:
    
    def __init__(self):
        
        self.DATA_ROOT = ''
        self.DATA_PATH = self.DATA_ROOT + 'GQA_vi'
        
        self.FEATS_PATH = {
            'default-frcn' : self.DATA_PATH + 'feats' + 'gqa-frcn',
            'default-grid' : self.DATA_PATH + 'feats' + 'gqa-grid'
        }
        
        self.RAW_PATH = {
            'train' : '',
            'val'   : '',
            'test'  : ''
        }
        
        self.RESULT_PATH = '.results/result_test'
        self.PRED_PATH  = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'
        
        make_dir(dir = 'result_test', path_dir = './results')
        make_dir(dir = 'pred', path_dir = './results')
        make_dir(dir = 'cache', path_dir = './results')
        make_dir(dir = 'log', path_dir = './results')
        make_dir(dir = 'ckpts', path_dir = '.')
        
    
    def check_path(self, dataset = None):
        print("[INFO] Checking dataset .........")
        
        if dataset:
            for item in self.FEATS_PATH.keys():
                if not os.path.exists(self.FEATS_PATH[item]):
                    print(self.FEATS_PATH[item], 'NOT EXIST')
                    exit(-1)
                    
            for item in self.RAW_PATH.keys():
                if not os.path.exists(self.RAW_PATH[item]):
                    print(self.RAW_PATH[item], 'NOT EXIST')
                    exit(-1)
                    
        print('[INFO] Finished!!!!!!!!!!')
        
        
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None
        
        self.data_size = None 
        self.token_size = None
        self.ans_size = None
        
    def load_ques_ans(self, idx):
        raise NotImplementedError()
    
    def load_img_feats(self, idx, iid):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)
        
        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)
        
        return torch.from_numpy(frcn_feat_iter), torch.from_numpy(grid_feat_iter), torch.from_numpy(bbox_feat_iter), torch.from_numpy(ques_ix_iter), torch.from_numpy(ans_iter)
    
    def __len__(self):
        return self.data_size
    
    def shuffle_list(self, list):
        random.shuffle(list)
        
        
class Adapter(torch.nn.Module):
    
    def __init__(self, __C):
        super(Adapter, self).__init__()
        self.__C = __C
        self.dataset_init()
        
    def dataset_init(self):
        raise NotImplementedError()
    
    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(frcn_feat, grid_feat, bbox_feat)
        
        return self.dataset_forward(feat_dict)
    
    def dataset_forward(self, feat_dict):
        raise NotImplementedError()
    
        
    
        