import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import sys
import os 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from core import BaseAdapter
from core.utils import make_mask, feat_filter
from core.ops import FC, LayerNorm, MLP


class BUTDAdapter(BaseAdapter):
    
    def __init__(self, __C):
        
        super(BUTDAdapter, self).__init__()
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
    
    def __init__(self, dims, activation = 'ELU', dropout_r = 0.):
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
    
class TopDownAttentionMap(nn.Module):
    
    def __init__(self, __C):
        super(TopDownAttentionMap, self).__init__()
        self.__C = __C
        self.linear_q = weight_norm(nn.Linear(self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE), dim = None)
        self.linear_v = weight_norm(nn.Linear(self.__C.IMG_FEAT_SIZE, self.__C.IMG_FEAT_SIZE), dim = None)
        self.non_linear = WeightNormMLP(
            dims = [self.__C.IMG_FEAT_SIZE + self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE],
            dropout_r = self.__C.DROPOUT_R
        )
        self.linear = weight_norm(nn.Linear(self.__C.HIDDEN_SIZE, 1), dim = None)
        
    def logits(self, q, v):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.non_linear(vq)
        logits - self.linear(joint_repr)
        return logits
        
        
    def forward(self, q, v):
        v = self.linear_v(v)
        q = self.linear_q(q)
        logits = self.logits(q, v)
        w = F.softmax(logits, 1)
        return w  
    
class AttendedJointMap(nn.Module): 
    
    def __init__(self, __C):
        super(AttendedJointMap, self).__init__()
        self.__C = __C
        self.v_att = TopDownAttentionMap(self.__C)
        self.q_net = WeightNormMLP(dims = [self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE])
        self.v_net = WeightNormMLP(dims = [self.__C.IMG_FEAT_SIZE, self.__C.HIDDEN_SIZE])
        
    def forward(self, q, v):
        att = self.v_att(q, v)
        att_v = (att * v).sum(1)
        q_repr = self.q_net(q)
        v_repr = self.v_net(att_v)
        joint_repr = q_repr * v_repr
        return joint_repr 
    
class BUTD(nn.Module): 
    
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(BUTD, self).__init__()
        self.__C = __C
        
        self.embedding = nn.Embedding(
            num_embeddings = token_size,
            embedding_dim  = self.__C.WORD_EMBED_SIZE
        )      
        
        if self.__C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        
        self.rnn = nn.LSTM(
            input_size = self.__C.WORD_EMBED_SIZE,
            hidden_size = self.__C.HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True
        )
        
        self.adapter = BUTDAdapter(self.__C)
        self.backbone = AttendedJointMap(self.__C)
        
        layers = [
            weight_norm(nn.Linear(self.__C.HIDDEN_SIZE, self.__C.FLAT_OUT_SIZE), dim = None), 
            nn.ReLU(),
            nn.Dropout(self.__C.CLASSIFER_DROPOUT_R, inplace = True),
            weight_norm(nn.Linear(self.__C.FLAT_OUT_SIZE, answer_size), dim = None)
        ]
        
        self.classifer = nn.Sequential(*layers)
        
    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)
        img_feat, _  = self.adapter(frcn_feat, grid_feat, bbox_feat)
        
        joint_feat = self.backbone(
            lang_feat[:, -1],
            img_feat
        )
        
        proj_feat = self.classifer(joint_feat)
        return proj_feat
    
