import torch
from core.datasets import Adapter
from core.layers import MCALayers
from core.layers import AttFlatten
from core.layers import LayerNorm
from utils.utils import make_mask

class Model(torch.nn.Module):
    
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Model, self).__init__()
        self.__C = __C
        
        self.embedding = torch.nn.Embedding(num_embeddings = token_size, embedding_dim= self.__C.WORD_EMBED_SIZE)
        
        self.lstm = torch.nn.LSTM(
            input_size = self.__C.WORD_EMBED_SIZE,
            hidden_size = self.__C.HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True
        )
        
        self.adapter = Adapter(self.__C)
        self.backbone = MCALayers(self.__C)
        
        self.attflatten_img = AttFlatten(self.__C)
        self.attflatten_lang = AttFlatten(self.__C)
        
        self.proj_norm = LayerNorm(self.__C.FLAT_OUT_SIZE)
        self.proj      = torch.nn.Linear(self.__C.FLAT_OUT_SIZE, answer_size)
        
    
    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        
        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        
        lang_feat, img_feat = self.backbone(lang_feat, img_feat, lang_feat_mask, img_feat_mask)
        
        lang_feat = self.attflatten_lang(lang_feat, lang_feat_mask)
        
        img_feat  = self.attflatten_img(img_feat, img_feat_mask)
        
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat        