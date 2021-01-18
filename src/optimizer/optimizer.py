import torch


class Optimizer(object):
    
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        
    
    def step(self):
        self._step += 1
        rate = self.rate()
        
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    
    def rate(self, step = None):
        if step is None:
            step = self._step
            
        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1.0 / 4
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2.0 / 4
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3.0 / 4
        else:
            r = self.lr_base
        return r
    

def optim(__C, model, data_size, lr_base = None):
    if lr_base is None:
        lr_base = __C.LR_BASE
        
    eval_str = 'param, lr=0'
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    
    return Optimizer(
        lr_base,
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = 0,
            betas = __C.OPT_BETAS,
            eps = __C.OPT_EPS
        ),
        data_size,
        __C.BATCH_SIZE
    )

def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r