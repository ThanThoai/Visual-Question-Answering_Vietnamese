import numpy as np
import glob 
import json
import re
import torch
import en_vectors_web_lg
from utils.preprocesstext import prep_ans

from core.datasets import Dataset


class DataLoader(Dataset):
    def __init__(self, __C):
        super(DataLoader, self).__init__()
        self.__C = __C
        print('')
        print("=====================================")
        print("[INFO] Loading dataset..............")
        ques_dict_preread = {
            'train' : json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'rb')),
            'val'   : json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'rb')),
            'test'  : json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'rb'))
        }
        
        frcn_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-frcn'] + '/*.npz')
        grid_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['default-grid'] + '/*.npz')
        
        self.ques_dict = {}
        split_list = self.__C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ques_dict_preread:
                self.ques_dict = {
                    **self.ques_dict, 
                    **ques_dict_preread[split]
                }
            else:
                self.ques_dict = {
                    **self.ques_dict,
                    **json.load(open(self.__C.RAW_PATH[self.__C.DATASET][split], 'rb'))
                }
        
        self.data_size = self.ques_dict.__len__()
        
        print("[INFO] Dataset size: {0}".format(self.data_size))
        
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)
        self.qid_list = list(self.ques_dict.keys())
        
        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize('', self.__C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print("[INFO] Quesion token vocab size: {0}".format(self.token_size))
        
        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token
        
        print("[INFO] Max token length: {0}, Trimmed to: {1}".format(self.max_token, self.max_token))
        
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('')
        self.ans_size = self.ans_to_ix.__len__()
        print("[INFO] Answer token vocab size: {0}".format(self.ans_size))
        print("Finished!!")
        print("=====================================")
        print("")
        
    def img_feat_path_load(self, path_list):
        iid_to_path = {}
        for ix, path in enumerate(path_list):
            iid = path.split('/')[-1].split('.')[0]
            iid_to_path[iid] = path
        return iid_to_path
        
    def tokenize(self, json_file, use_glove):
        token_to_ix, max_token = json.load(open(json_file, 'rb'))[2:]
        spacy_tool = None
        
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            
        pretrained_emb = []
        for word in token_to_ix:
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)
        pretrained_emb = np.array(pretrained_emb)
        return token_to_ix, pretrained_emb, max_token
    
    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'rb'))[:2]
        return ans_to_ix, ix_to_ans
    
    def load_ques_ans(self, idx):
        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']
        
        ques = self.ques_dict[qid]['question']
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token = self.max_token)
        ans_iter = np.zeros(1)
        
        if self.__C.RUN_MODE in ['train']:
            ans = self.ques_dict[qid]['answer']
            ans_iter = self.proc_ans(ans, self.ans_to_ix)
            
        return ques_ix_iter, ans_iter, iid
    
    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_iter = self.proc_img_feat(frcn_feat['x'], img_feat_pad_size = self.__C.FEAT_SIZE['FRCN_FEAT_SIZE'][0])
        
        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat_iter = grid_feat['x']
        
        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'], 
                (frcn_feat['height'], frcn_feat['width'])
            ),
            img_feat_pad_size = self.__C.FEAT_SIZE['BBOX_FEAT_SIZE'][0]
        )
        
        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter
    
    def proc_img_feat(self, img_feat, img_feat_pad_size):
        
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]
            
        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode = 'constant',
            constant_values=0
        )
        
        return img_feat
    
    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype = np.float32)
        
        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])
        return bbox_feat
    
    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        
        words = re.sub(r"([.,!?''()*#:;])", '', ques.lower()).replace('-', '').replace('/', ' ').split()
        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']
            if ix + 1 == max_token:
                break
        
        return ques_ix
    
    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans = prep_ans(ans)
        ans_ix[0] = ans_to_ix[ans]
        return ans_ix