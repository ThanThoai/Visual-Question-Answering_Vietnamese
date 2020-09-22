import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import math 


class FC(nn.Module):
    """
    Fully Connected
    """
    
    def __init__(self, in_size, out_size, dropout_r = 0., use_relu = True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        
        self.linear  = nn.Linear(in_size, out_size)
        
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout= nn.Dropout(dropout_r)
            
    
    def forward(self, x):
        
        x = self.linear(x)
        if self.relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x
    
class MLP(nn.Module):
    """
    Multi-layers Perceptrons
    """
    def __init__(self, in_size, hid_size, out_size, dropout_r = 0., use_relu = True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, hid_size, dropout_r = dropout_r, use_relu = use_relu)
        self.linear = nn.Linear(hid_size, out_size)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.linear(x)
        return x
    
class LayerNorm(nn.Module):
    
    def __init__(self, size, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps 
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std  = x.std(-1, keepdim = True)
        
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
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
        
    def forward(self, x, y, x_mask, y_mask):
        
        for e in self.encoder_list:
            x = e(x, x_mask)
        
        for d in self.decoder_list:
            y = d(y, x, y_mask, x_mask)
        
        return x, y