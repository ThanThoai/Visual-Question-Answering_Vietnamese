  
import torch
from .MLP import MLP

class AttFlatten(torch.nn.Module):
    
    def __init__(self, __C):
        super(AttFlatten, self).__init__()
        self__C = __C
        
        self.mlp = MLP(
            in_size = self.__C.HIDDEN_SIZE,
            hidden_size = self.__C.FLAT_MLP_SIZE,
            out_size  = self.__C.FLAT_GLIMPSES,
            dropout_r = self.DROPOUT_R,
            use_relu  = True
        )
        
        self.linear_merge = torch.nn.Linear(self.__C.HIDDEN_SIZE * self.__C.FLAT_GLIMPSES, self.__C.FLAT_OUT_SIZE)
        
    def forward(self, x, x_mask):
        
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = torch.nn.functional.softmax(att, dim = 1)
        
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i:i+1] * x), dim = 1)
        x_atted = torch.cat(att_list, dim = 1)
        x_atted = self.linear_merge(x_atted)
        
        return x_atted  