import torch 
import math

import torch 
import math 
from .FC import FC
from .MLP import MLP
from .LayerNorm import LayerNorm


class MultiHeadAtt(torch.nn.Module):
    
    def __init__(self, __C):
        super(MultiHeadAtt, self).__init__()
        self.__C = __C
        
        self.linear_v = torch.nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = torch.nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = torch.nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        
        self.linear_merge = torch.nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        
        self.dropout = torch.nn.Dropout(__C.DROPOUT_R)
        
    def forward(self, value, key, query, mask):
        
        n_batches = query.size(0)
        value = self.linear_v(value).view(n_batches,
                                          -1, 
                                          self.__C.MULTI_HEAD,
                                          int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
                                          ).transpose(1, 2)
        
        key = self.linear_k(key).view(n_batches,
                                          -1, 
                                          self.__C.MULTI_HEAD,
                                          int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
                                          ).transpose(1, 2)
        
        query = self.linear_q(query).view(n_batches,
                                          -1, 
                                          self.__C.MULTI_HEAD,
                                          int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
                                          ).transpose(1, 2)
        
        atted = self.att(value, key, query, mask)
        
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches, 
            -1, 
            self.__C.HIDDEN_SIZE
        )
        atted = self.linear_merge(atted)
        
        return atted
    
    def att(self, value, key, query, mask):
        
        dim_key = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            
        att_map = torch.nn.functional.softmax(scores, dim = -1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)
        