import torch
from .FC import FC

class MLP(torch.nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size, dropout_r = 0., use_relu = True):
        super(MLP, self).__init__()
        self.fc = FC(in_size = in_size, out_size = hidden_size, dropout_r = dropout_r, use_relu = use_relu)
        self.linear = torch.nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        return self.linear(self.fc(x))
        