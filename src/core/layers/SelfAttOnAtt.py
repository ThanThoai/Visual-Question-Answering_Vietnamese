import torch
from .FFN import FNN
from .MultiHeadAtt import MultiHeadAtt
from .LayerNorm import LayerNorm



class SelfAttOnAtt(torch.nn.Module):

    def __init__(self, __C, dim, dim_head = 64, heads = 8):
        super(SelfAttOnAtt, self).__init__()
        inner_dim = dim_head * heads
        self.mhatt = MultiHeadAtt(__C)
        self.ffn = FNN(__C)
        self.dropout_1 = torch.nn.Dropout(__C.DROPOUT_R)
        self.layernorm_1 = LayerNorm(__C)

        self.dropout_2 = torch.nn.Dropout(__C.DROPOUT_R)
        self.layernorm_2 = LayerNorm(__C)

        self.to_kv = torch.nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = torch.nn.Linear(dim, inner_dim, bias = False)

        self.linear = torch.nn.Linear(2 * inner_dim, 2 * inner_dim)
        self.activation = torch.nn.GLU()


    def forward(self, y, y_mask):

        to_kv = self.mhatt(y, y, y, y_mask)
