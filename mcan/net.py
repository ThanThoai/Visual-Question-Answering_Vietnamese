from mcan.mca import MCA_ED, LayerNorm, FC, MLP
from mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu = True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted



class Net(nn.Module):
    def __init__(self, __C, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.adapter = Adapter(__C)
        self.backbone = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, lang_feat, ques_idx):

        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat
