import os
import torch
import random
from types import MethodType
import numpy as np

class Cfgs(object):

    def __init__(self):

        self.DATASET_PATH = './dataset/text'
        self.FEATURE_PATH = './dataset/image'

        self.IMG_FEAT_PATH = {
            "train" : os.path.join(self.FEATURE_PATH, 'train'),
            "val" : os.path.join(self.FEATURE_PATH, 'val'),
            "test" : os.path.join(self.FEATURE_PATH, 'test')
        }

        self.QUESTION_PATH = {
            'train' : os.path.join(self.DATASET_PATH, "train_question_vi.json"),
            'test' : os.path.join(self.DATASET_PATH, "test_question_vi.json"),
            "val"  : os.path.join(self.DATASET_PATH, 'val_question_vi.json')
        }

        self.ANSWER_PATH = {
            'train' : os.path.join(self.DATASET_PATH, "train_annotations.json"),
            "val"   : os.path.join(self.DATASET_PATH, "val_annotations.json")
        }

        self.RESULT_PATH = "./result/test"
        self.PRED_PATH  = "./result/predictions"
        self.CACHE_PATH = "./result/cache"

        self.LOG_PATH  = "./logs"
        self.CHECKPOINT_PATH = "./checkpoint"

        print("Check folder exists!!!!")

        if not os.path.isdir("./result"): 
            os.mkdir("./result")
        
        if "test" not in os.listdir("./result"):
            os.mkdir(os.path.join("./result", "test"))
        
        if "predictions" not in os.listdir("./result"):
            os.mkdir(os.path.join("./result", "predictions"))

        if "cache" not in os.listdir("./result"):
            os.mkdir(os.path.join("./result", "cache"))
        
        if not os.path.isdir(self.LOG_PATH):
            os.mkdir(self.LOG_PATH)

        if not os.path.isdir(self.CHECKPOINT_PATH): 
            os.mkdir(self.CHECKPOINT_PATH)

        print("Done!!!!")

        ##Version-control

        self.SEED = random.randint(0, 999999)
        self.VERSION = str(self.SEED)
        self.RESUME = False
        self.CHECKPOINT = self.VERSION
        self.CHECKPOINT_EPOCH = 0
        self.MAX_EPOCH = 13
        self.VERBOSE = True

        self.FOLLOW_CONTROL = 'train-val'
        self.RUN_MODE = 'train'
        self.EVAL_EVERY_EPOCH = True
        self.TEST_SAVE_PRED = False

        self.PRELOAD = False
        self.BERT_MODEL = None
        self.MCAN_MODEL = "Encode-Decode"
        if self.BERT_MODEL == 'BASE':
            self.WORD_EMBED_SIZE = 768
        elif self.BERT_MODEL == 'LARGE':
            self.WORD_EMBED_SIZE = 1024
        else:
            print(self.BERT_MODEL + "not supported")
            exit(-1)

        self.MAX_TOKEN = 14
        self.IMG_FEAT_PAD_SIZE = 100
        self.TEXT_FEAT_PAD_SIZE = 20
        self.BATCH_SIZE = 64

        self.NUM_WORKERS = 8
        self.PIN_MEMORY = True
        self.GRAD_ACCU_STEPS = 1

        self.LAYERS = 6
        self.HIDDEN_SIZE = 512
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512 
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024


        self.LR_BASE = 0.0001
        self.LR_DECAY_R = 0.2
        self.LR_DECAY_LIST = [10, 12]
         
        self.GRAD_NORM_CLIP = -1
        self.OPT_BETAS = (0.9, 0.98)
        self.OPT_EPS = 1e-9

    

    def check_dataset(self):

        print("Checking dataset....")

        for mode in self.IMG_FEAT_PATH.keys():
            assert os.path.exists(self.IMG_FEAT_PATH[mode]), self.IMG_FEAT_PATH[mode] + " NOT FOUND"
        
        for mode in self.QUESTION_PATH.keys():
            assert os.path.exists(self.QUESTION_PATH[mode]), self.QUESTION_PATH[mode] + " NOT FOUND"
        
        for mode in self.ANSWER_PATH.keys():
            assert os.path.exists(self.ANSWER_PATH[mode]), self.ANSWER_PATH[mode] + " NOT FOUND"
        
        print("Done!!!!")

    def parse_to_dict(self, args):
        args_dict = dict()
        for arg in dir(args):
            if not arg.startswith("_") and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    
    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def process(self):

        os.environ["CUDA_VISIBLE_DEVICES"] =  '0'
        torch.set_num_threads(2)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        

        np.random.seed(self.SEED)
        random.seed(self.SEED)

        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)

        self.FF_SIZE = int(self.HIDDEN_SIZE * 4)
        assert self.HIDDEN_SIZE % self.MULTI_HEAD == 0
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE / self.MULTI_HEAD)

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not isinstance(getattr(self, attr), MethodType):
                print("{%-17s} -> " %attr, getattr(self, attr))
        return ''