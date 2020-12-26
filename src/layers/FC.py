import torch

class FC(torch.nn.Module):
    
    def __init__(self, in_size, out_size, dropout_r = 0., use_relu = True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        
        self.linear = torch.nn.Linear(in_size, out_size)
        
        if use_relu:
            self.relu = torch.nn.ReLU(in_size)
        if dropout_r > 0:
            self.dropout = torch.nn.dropout(self.dropout_r)
            
    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
            
        return x 