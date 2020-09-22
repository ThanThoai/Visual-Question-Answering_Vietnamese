from .core import * 
import torch.nn as nn
import torch.nn.functional as F
import torch


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
    
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim = -1
        ) == 0).unsqueeze(1).unsqueeze(2)    
        
    def forward(self, img_feat, ques_ix):
        
        pass     