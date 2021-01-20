import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(nn.Module):
    def __init__(self, __C):
        super(Adapter, self).__init__()
        self.data_init(__C)

    @staticmethod
    def make_mask(feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def data_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

        if __C.USE_AUX_FEAT:
            self.grid_linear = nn.Linear(__C.FEAT_SIZE['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = {
            "FRCN_FEAT" : frcn_feat,
            "GRID_FEAT" : grid_feat,
            "BBOX_FEAT" : bbox_feat
        }
        return data_forward(feat_dict)


    def data_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        return img_feat, img_feat_mask
