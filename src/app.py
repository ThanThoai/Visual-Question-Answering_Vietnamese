import os
import copy
from dataloader import DataLoader
import torch
import os
import datetime
import time
import shutil
import numpy as np
from core.optimizer import optim, adjust_lr
from model import Model


class App():
    
    def __init__(self, __C, dataset):
        self.__C = __C
        self.dataset = dataset
    
    def train(self):
        
        model = Model(
            self.__C, 
            self.dataset.pretrained_emb, 
            self.dataset.token_size, 
            self.dataset.ans_size
        )
        
        model.cuda()
        model.train()
        
        loss_fn = eval('torch.nn.' + self.__C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION +"').cuda()")
        
        if self.__C.RESUME:
            print("[INFO Resume training---------")
            
            if self.__C.CKPT_PATH is not None:
                print("[WARRING] Using CKPT_PATH args, 'CKPT_VERSION' and CKPT_EPOCH will not work")
                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPT_PATH + '/ckpt_' + self.__C.CKPT_VERSION + '/epoch' + str(self.__C.CKPT_EPOCH)  + '.pkl'
                
            print("[INFO] Loading ckpt from {0}".format(path))
            ckpt = torch.load(path)
            print("[INFO] Finish!!!") 
            
            model.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt['epoch']
            
            optimizer = optim(self.__C, model, self.dataset.data_size, ckpt['lr_base'])
            optimizer._step = int(self.dataset.data_size / self.__C.BATCH_SIZE * start_epoch) 
            optimizer.optimizer.load_state_dict(ckpt['optimizer'])
        
            if ('ckpt_' + self.__C.VERSION) not in os.listdir(self.__C.CKPT_PATH):
                os.mkdir(self.__.CKPT_PATH + '/ckpt_' + self.__C.VERSION)
                
        else:
            
            if ('ckpt_' + self.__C.VERSION) not in os.listdir(self.__C.CKPT_PATH):
                os.mkdir(self.__C.CKPT_PATH + '/ckpt' + self.__C.VERSION)
            
            optimizer = optim(self.__C, model, self.dataset.data_size)
            start_epoch = 0
        
        loss_sum = 0
        named_param = list(model.named_parameters())
        grad_norm = np.zeros(len(named_param))
        
        dataloader = DataLoader(
            self.dataset,
            batch_size = self.__C.BATCH_SIZE,
            shuffle = True, 
            num_workers = self.__C.NUM_WORKERS,
            pin_memory = self.__C.PIN_MEMORY,
            drop_last = True
        )
        
        log_file = open(
            self.__C.LOG_PATH + '/log_run_' + self.__C.VERSION + '.txt', 'a+'
        )
        log_file.write(str(self.__C))
        log_file.close()
        
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            log_file = open(self.__C.LOG_PATH + '/log_run_' + self.__C.VERSION + '.txt', 'a+')
            log_file.write("[INFO] Time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
            log_file.close()
            
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optimizer, self.__C.LR_DECAY_R)
                
            time_start = time.time()
            #TODO 