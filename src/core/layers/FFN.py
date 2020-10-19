import torch 
from .MLP import MLP 


class FFN(torch.nn.Module):
    
    def __init__(self, __C):
        super(FFN, self).__init__()
        
        self.mlp = MLP(
            in_size = __C.HIDDEN_SIZE,
            hidden_size = __C.FF_SIZE,
            out_size  = __C.HIDDEN_SIZE,
            dropout_r = __C.DROPOUT_R,
            use_relu  = True
        )
        
    def forward(self, x):
        return self.mlp(x)