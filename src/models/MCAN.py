import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from src.core import BaseAdapter
from src.core.utils import *
from src.core.ops import *


class MCANAdapter(BaseAdapter):
    
    def __init__(self, __C):
        super(MCANAdapter, self).__init__(__C)
        self.__C = __C
    
    
    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        return torch.cat((bbox, area.unsqueeze(2)), -1)
        
    def dataset_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['FRCM_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        
    def dataset_forward(self, feat_dict):
        frcn_feat = feat_dict['FastRCNN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        
        img_feat_mask = make_mask(frcn_feat)
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim = -1)
        img_feat = self.frcn_linear(frcn_feat)
        return img_feat, img_feat_mask        
    
class MultiHeadAtt(nn.Module):
    """Multi-Head Attention""" 
    
    def __init__(self, __C):
        """
        @param _C : config 
        """
        super(MultiHeadAtt, self).__init__()
        
        self.__C = __C
        #value
        self.linear_v = nn.linear(self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE)
        #key
        self.linear_k = nn.linear(self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE)
        #query
        self.linear_q = nn.linear(self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE)     
        #merge 
        self.linear_merge = nn.linear(self.__C.HIDDEN_SIZE, self.__C.HIDDEN_SIZE)
        
        self.dropout = nn.Dropout(self.__C.DROPOUT_R)
        
    def attention(self, value, key, query, mask):
        d_k = query.size(-1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim = -1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)
        
    def forward(self, v, k, q, mask):
        
        n_batches = q.size(0)
        
        v = self.linear_v(v).view(
            n_batches, 
            -1, 
            self.__C.MULTI_HEAD, 
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        
        k = self.linear_k(k).view(
            n_batches, 
            -1,
            self.__C.MULTI_HEAD, 
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        
        q = self.linear_q(q).view(
            n_batches, 
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        
        attented = self.attention(v, k, q, mask)
        attented = attented.transpose(1, 2).contiguous().view(
            n_batches,  
            -1,
            self.__C.HIDDEN_SIZE
        )
        
        attented = self.linear_merge(attented)
        return attented
        
class FFN(nn.Module):
    """Fead Forward Nets""" 
    
    def __init__(self, __C):
        super(FFN, self).__init__()
        
        self.mlp = MLP(
            in_size = __C.HIDDEN_SIZE, 
            hid_size = __C.FF_SIZE,
            out_size = __C.HIDDEN_SIZE, 
            dropout_r = __C.DROPOUT_R
        )
        
    def forward(self, x):
        return self.mlp(x)

class SelfAttention(nn.Module):
    
    def __init__(self, __C):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAtt(__C)
        self.ffn = FFN(__C)
        self.dropout_1 = nn.Dropout(__C.DROPOUT_R)
        self.norm_1    = LayerNorm(__C.HIDDEN_SIZE)
        self.dropout_2 = nn.Dropout(__C.DROPOUT_R)
        self.norm_2    = LayerNorm(__C.HIDDEN_SIZE)
        
    def forward(self, x, x_mask):
        x = self.norm_1(x + self.dropout_1(self.mha(x, x, x, x_mask)))        
      
        x = self.norm_2(x + self.dropout_2(self.ffn(x)))
        
        return x 

class SelfGuildedAttention(nn.Module):
    
    def __init__(self, __C):
        super(SelfGuildedAttention, self).__init__()
        self.mha_1 = MultiHeadAtt(__C)
        self.mha_2 = MultiHeadAtt(__C)
        self.ffn = FFN(__C)
        
        self.dropout_1 = nn.Dropout(__C.DROPOUT_R)
        self.norm_1    = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout_2 = nn.Dropout(__C.DROPOUT_R)
        self.norm_2    = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout_3 = nn.Dropout(__C.DROPOUT_R)
        self.norm_3    = LayerNorm(__C.HIDDEN_SIZE)
        
    
    def forward(self, x, y, x_mask, y_mask):
        x = self.norm_1(x + self.dropout_1(self.mha_1(x, x, x, x_mask)))
        
        x = self.norm_2(x + self.dropout_2(self.mha_2(y, y, x, y_mask)))
        
        x = self.norm_3(x + self.dropout_3(self.ffn(x)))
        return x
    

class MCA_ED(nn.Module):
    
    """MAC Layers Cascaded by Encoder-Decoder""" 
    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        
        self.encoder_list = nn.ModuleList([SelfAttention(__C) for _ in range(__C.LAYER)])
        self.decoder_list = nn.ModuleList([SelfGuildedAttention(__C) for _ in range(__C.LAYER)])
        
    def forward(self, y, x, y_mask, x_mask):
        
        for e in self.encoder_list:
            y = e(y, y_mask)
        
        for d in self.decoder_list:
            x = d(x, y, x_mask, y_mask)
        
        return x, y
    

class AttentionFlatten(nn.Module):

    def __init__(self, __C):
        super(AttentionFlatten, self).__init__()
        self.__C = __C
        
        self.mlp = MLP(
            in_size = self.__C.HIDDEN_SIZE, 
            hid_size = self.__C.FLAT_MLP_SIZE,
            out_size = self.__C.FLAT_GLIMPSES,
            dropout_r = self.__C.DROPOUT_R
        )
        
        self.linear_merge = nn.Linear(
            self.__C.HIDDEN_SIZE * self.__C.FLAT_GLIMPSES,
            self.__C.FLAT_OUT_SIZE
        )
        
    def forward(self, x, x_mask):
        attention = self.mlp(x)
        attention = attention.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        
        attention = F.softmax(attention, dim = 1)
        attention_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            attention_list.append(torch.sum(attention[:, :, i: i + 1] * x, dim =1))
        x_attention = torch.cat(attention_list, dim = 1)
        x_attention = self.linear_merge(x_attention)
        return x_attention
    
    
class MCAN(nn.Module):
    
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(MCAN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings = token_size,
            embedding_dim  = __C.WORD_EMBED_SIZE
        )
        
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.lstm = nn.LSTM(
            input_size = __C.WORD_EMBED_SIZE,
            hidden_size = __C.HIDDEN_SIZE, 
            num_layers = 1,
            batch_first = True
        )
        
        self.img_feat_linear = nn.Linear(
            in_features = __C.IMG_FEAT_SIZE,
            out_features = __C.HIDDEN_SIZE
        )
        
        self.backbone = MCA_ED(__C)
        self.attention_flatten_img = AttentionFlatten(__C)
        self.attention_flatten_text = AttentionFlatten(__C) 
        
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj      = nn.Linear(
            in_features = __C.FLAT_OUT_SIZE,
            out_features = __C.answer_size
        )   
        
    def forward(self, img_feat, ques_ix):
        
        pass     
    