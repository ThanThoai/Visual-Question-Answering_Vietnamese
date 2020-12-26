import torch
from .SelfAtt import SelfAtt
from .SelfGuidedAtt import SelfGuidedAtt


#Encoder-Decoder
class MCALayers(torch.nn.Module):
    
    def __init__(self, __C):
        super(MCALayers, self).__init__()
        self.enc_list = torch.nn.ModuleList([SelfAtt(__C) for _ in range(__C.LAYER)])
        self.dec_sa_list = torch.nn.ModuleList([selfAtt(__C) for _ in range(__C.LAYER)])
        self.dec_ga_list = torch.nn.ModuleList([SelfGuidedAtt(__C) for _ in range(__C.LAYER)])
        
    
    def forward(self, y, x, y_mask, x_mask):
        
        for enc in self.enc_list:
            y = enc(y, y_mask)
            
        for i in range(len(self.dec_sa_list)):
            x = self.dec_sa_list[i](x, x_mask)
            x = self.dec_ga_list[i](x = x, y = y, x_mask = x_mask, y_mask = y_mask)
            
        return y, x


#Stack
class MCALayerStack(torch.nn.Module):

    def __init__(self, __C):

        super(MCALayerStack, self).__init__()

        self.enc_list = torch.nn.ModuleList([SelfAtt(__C) for _ in range(__C.LAYER)])
        self.dec_sa_list = torch.nn.ModuleList([SelfAtt(__C) for _ in range(__C.LAYER)])
        self.dec_ga_list = torch.nn.ModuleList([SelfGuided(__C) for _ in range(__C.LAYER)])


    def forward(self, x, y, x_mask, y_mask):
        for i in range(len(self.enc_list)):
            y = self.enc_list[i](y, y_mask)
            x = self.dec_sa_list[i](x, x_mask)
            x = self.dec_ga_list[i](x = x, y = y, x_mask = x_mask, y_mask = y_mask)

        return y, x



