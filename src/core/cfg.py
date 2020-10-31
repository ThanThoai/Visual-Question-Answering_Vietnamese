import os 
import torch
import random
import numpy as np 
from types import MethodType
from datasets import PATH 


class Config(PATH):
    
    def __init__(self):
        super(Config, self).__init__()
        
        self.GPU = '0'
        self.SEED = random.randint(0, 999999)
        
        self.VERSION = str(self.SEED)
        self.RESUME = False 
        self.CKPT_VERSION = self.VERSION
        self.CKPT_EPOCH = 0 
        self.CKPT_PATH = None
        self.VERBOSE = True 
        
        
        self.MODEL = ''
        self.MODEL_USE = ''
        self.DATASET = ''
        
        self.RUN_MODE = ''
        self.EVAL_EVERY_EPOCH = True 
        self.TEST_SAVE_PRED = False 
        
        self.TRAIN_SPLIT = 'train'
        self.EMBED_METHOD = '' 
        self.WORD_EMBED_SIZE = None
        self.FEAT_SIZE = {
            'FRCN_FEAT_SIZE' : (100, 2048),
            'GRID_FEAT_SIZE' : (49, 2048),
            'BBOX_FEAT_SIZE' : (100, 5)
        }   
        
        self.BBOX_NORMALIZE = False
        self.BATCH_SIZE = 64
        self.NUM_WORKERS = 8 
        
        self.PIN_MEM = True
        self.GRAD_ACCU_STEPS = 1
        
        
        self.LOSS_FUNC = ''
        self.LOSS_REDUCTION = ''
        self.LR_BASE = 0.0001
        self.LR_DECAY_R = 0.2
        self.LR_DECAY_LIST = [10, 12]
        self.WARMUP_EPOCH = 3
        self.MAX_EPOCH = 13
        
        self.GRAD_NORM_CLIP = -1
        self.OPT = ''
        self.OPT_PARAMS = {}
        
    def str_to_bool(self, args):
        
        bool_list = ['EVAL_EVERY_EPOCH', 'TEST_SAVE_PRED', 'RESUME', 'PIN_MEM', 'VERBOSE', 'EMBED_METHOD']
        
        for arg in dir(args):
            if arg in bool_list and getattr(args, arg) is not None:
                setattr(args, arg, eval(getattr(args, arg)))
        return args
    
    
    def parse_to_dict(self, args):
        args_dict = {}
        for arg in args_dict:
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict
    
    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])
            
    def proc(self):
        assert self.RUN_MODE in ['TRAIN', 'VAL', 'TEST']
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)
        
        self.check_path(self.DATASET)
        
        torch.manual_seed(self.SEED)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.SEED)
        else:
            torch.cuda.manual_seed_all(self.SEED)
        
        torch.backends.cudnn.deterministic = True
        
        np.random.seed(self.SEED)
        random.seed(self.SEED)

        if self.CKPT_PATH is not None:
            print("[WARRING] You are now using 'CKPT_PATH' args")
            print("[WARRING] 'CKPT_VERSION' and 'CKPT_EPOCH' will not work")
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(random.randit(0, 9999999))
                    
                    
        
        self.SPLIT = self.SPLITS[self.DATASET]
        self.SPLIT['TRAIN'] = self.TRAIN_SPLIT
        if self.SPLIT['VAL'] in self.SPLIT['TRAIN'].split('+') or self.RUN_MODE not in ['TRAIN']:
            self.EVAL_EVERY_EPOCH = False
        if self.RUN_MODE not in ['TEST']:
            self.TEST_SAVE_PRED = False
        
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)
        
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)
        
        assert self.LOSS_FUNC in ['CE', 'BCE', 'KLD', 'MSE']
        assert self.LOSS_REDUCTION in ['NONE', 'ELEMENTWISE_MEAN', 'SUM']
        
        self.LOSS_FUNC_NAME_DICT = {
            'CE' : 'CrossEntropyLoss',
            'BCE': 'BCEWithLogitsLoss',
            'KLD': 'KLDivLoss',
            'MSE': 'MSELoss'
        }
        
        self.LOSS_FUNC_NONLINEAR = {
            'CE' : [None, 'flat'],
            'BCE': [None, None],
            'KLD': ['log_sofmax', None],
            'MSE': [None, None]
        }
        
        self.TASK_LOSS = {
            'GQA' : ['CE']
        }
        
        assert self.LOSS_FUNC in self.TASK_LOSS[self.DATASET], self.DATASET + 'task only support' + str(self.TASK_LOSS[self.DATASET]) + 'loss.' + 'Modify the LOSS_FUNC in configs to get better score'
        
        
        assert self.OPT in ['Adam', 'Adamax', 'RMSPprop', 'SGD', 'Adadelta', 'Adagrad']
        optim = getattr(torch.optim, self.OPT)
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[3:, optim.__init__.__code__.co_argcount], optim.__init__.__defaults__[1:]))
        
        def all(iterable):
            for element in iterable:
                if not element:
                    return False
            return True 
        
        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))
        
        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                print("[ERROR] To avoid ambiguity, set the value of 'OPT_PARAMS' to string type")
                exit(-1)
        self.OPT_PARAMS = {**default_params_dict, ** self.OPT_PARAMS}
        
    def __str__(self):
        __C_str = ''
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                __C_str += '{ %-17s }->' %attr + str(getattr(self, attr)) + '\n'
        return __C_str
    

if __name__ == '__main__':
    pass 