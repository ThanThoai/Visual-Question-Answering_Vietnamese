import torch


from .src.layers import MCALayersED, MCALayersStack
from .src.layers import AttFlatten
from .src.layers import LayerNorm

def make_mask(feature):
    return (torch.sum(torch.abs(feature), dim = -1) == 0).unsqueeze(1).unsqueeze(2)

class Model(torch.nn.Module):
    
    def __init__(self, __C, answer_size):
        super(Model, self).__init__()
        self.__C = __C
        self.backbone = None
        if self.__C.MCAN_MODEL == "Encode-Decode":
            self.backbone = MCALayersED(self.__C)
        elif self.__C.MCAN_MODEL == "Stack":
            self.backbone = MCALayersStack(self.__C)
        else:
            print(f"MODEL {self.__C.MCAN_MODEL} not supported")
            exit(-1)

        
        self.attflatten_img = AttFlatten(self.__C)
        self.attflatten_lang = AttFlatten(self.__C)
        
        self.proj_norm = LayerNorm(self.__C.FLAT_OUT_SIZE)
        self.proj      = torch.nn.Linear(self.__C.FLAT_OUT_SIZE, answer_size)
        
    
    def forward(self, img_feat, ques_ix, ques_feat):
        
        ques_feat_mask = make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = make_mask(img_feat)
        lang_feat, img_feat = self.backbone(ques_feat, img_feat, lang_feat_mask, img_feat_mask)
        
        lang_feat = self.attflatten_lang(lang_feat, ques_feat_mask)
        
        img_feat  = self.attflatten_img(img_feat, img_feat_mask)
        
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat      