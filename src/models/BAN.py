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
        self.v_net = WeightNormMLP(
            dims = [self.__C.IMG_FEAT_SIZE, self.__C.BA_HIDDEN_SIZE], 
            dropout_r = self.__C.DROPOUT_R
            )
        self.q_net = WeightNormMLP(
            dims = [self.__C.HIDDEN_SIZE, self.__C.BA_HIDDEN_SIZE],
            dropout_r = self.__C.DROPOUT_R
            )
        
        if not attention:
            self.p_net = nn.AvgPool(self.__C.K_TIMES, stride = self.__C.K_TIMES)
        else:
            self.dropout = nn.Dropout(self.__C.CLASSIFER_DROPOUT_R)
            self.h_mat = nn.Parameter(torch.Tensor(1, self.__C.GLIMPSE, 1, self.__C.BA_HIDDEN_SIZE).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, self.__C.GLIMPSE,1, 1).normal_())
        
    def forward(self, v, q):
        v_ = self.dropout(self.v_net(v))
        q_ = self.q_net(q) 
        
        logits = torch.einsum('xhyk,byk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        
        return logits

    def forward_weight(self, v, q, w):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        logits = logits.unsqueeze(1)
        logits = self.p_net(logits).squeeze(1) * self.__C.K_TIMES
        return logits
    

class BiAttention(nn.Module):

    def __init__(self, __C):
        super(BiAttention, self).__init__()
        self.__C = __C
        self.logits = weight_norm(
            BilinearConnect(
                __C = self.__C,
                attention = True
            ), 
            name = 'h_mat', 
            dim = None
        )
    
    def forward(self, v, q, v_mask = True, logit = False, mask_with = -float('inf')):
        
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)
        
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)
        
        if not logit:
            
            p = F.softmax(logits.view(-1, self.__C.GLIMPSE, v_num * q_num), 2)
            return p.view(-1, self.__C.GLIMPSE, v_num. q_num), logits
        
        return logits
    

class BiliearAttentionNetwork(nn.Module):
    
    def __init__(self, __C):
        super(BiliearAttentionNetwork, self).__init__()
        self.__C = __C
        
        self.BiAtt = BiAttention(self.__C)
        b_net = []
        q_prj = []
        c_prj = []
        
        for i in range(self.__C.GLIMPSE):
            
            b_net.append(BilinearConnect(self.__C))
            q_prj.append(WeightNormMLP(
                dims = [self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE],
                activation = '',
                dropout_r = self.__C.DROPOUT_R
            )) 
        self.b_net = nn.ModuleList(b_net) 
        self.q_prj = nn.ModuleList(q_prj)
        
    def forward(self, q, v):
        att, logits = self.BiAtt(v, q)
        for g in range(self.__C.GLIMPSE):
            bi_emb = self.b_net[g].forward_weight(
                v = v,
                q = q,
                w = att[:, g, :, :]
            )
            q = self.q_prj[g](bi_emb.unsqueeze(1)) + q
        return q  
    
    
class BAN(nn.Module):
    
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        
        super(BAN, self).__init__()
        self.__C = __C
        
        self.embedding = nn.Embedding(
            num_embeddings = token_size,
            embedding_dim =  self.__C.WORD_EMBED_SIZE
        )       
        
        if self.__C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        
        self.rnn = nn.GRU(
            input_size = self.__C.WORD_EMBED_SIZE,
            hidden_size = self.__C.HIDDEN_SIZE, 
            num_layers = 1,
            batch_first = True
        )
        
        self.adapter = BANAdapter(self.__C)
        self.backbone = BiliearAttentionNetwork(self.__C)
        
        layers = [
            weight_norm(nn.Linear(self.__C.HIDDEN_SIZE, self.__C.FLAT_OUT_SIZE), dim = None),
            nn.Relu(),
            nn.Dropout(self.__C.CLASSIFER_DROPOUT_R, inplace = True),
            weight_norm(nn.Linear(self.__C.FLAT_OUT_SIZE, answer_size), dim = None)
        ]
        
        self.classifer = nn.Sequential(*layers)
        
    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)
        
        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)
        lang_feat = self.backbone(lang_feat, img_feat)
        
        proj_feat = self.classifer(lang_feat.sum(1))
        return proj_feat