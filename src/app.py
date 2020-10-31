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
from core.eval import eval_model


class App():

    def __init__(self, __C):
        self.__C = __C
        self.dataset = DataLoader(__C)
        
    def run(self, run_mode, dataset):
        if run_mode == 'train': 
            if self.__C.RESUME is False:
                self.empty_log(self.__C.VERSION)
            self.train(self.__C)
        elif run_mode == 'val': 
            self.test(self.__C, dataset, validation = True)
        elif run_mode == 'test': 
            self.test(self.__C, dataset)
        else:
            exit(-1)
            

    def train(self, dataset_eval = None):

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
            for step, (frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter) in enumerate(dataloader):
                optimizer.zero_grad()
                frcn_feat_iter = frcn_feat_iter.cuda()
                grid_feat_iter = grid_feat_iter.cuda()
                bbox_feat_iter = bbox_feat_iter.cuda()
                ques_ix_iter   = ques_ix_iter.cuda()
                ans_iter  = ans_iter.cuda()

                loss_tmp = 0
                for accuracy_step in range(self.__C.GRAD_ACCU_STEPS):
                    loss_tmp = 0
                    sub_frcn_feat_iter = frcn_feat_iter[accuracy_step * self.__C.SUB_BATCH_SIZE: (accuracy_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_grid_feat_iter = grid_feat_iter[accuracy_step * self.__C.SUB_BATCH_SIZE: (accuracy_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_bbox_feat_iter = bbox_feat_iter[accuracy_step * self.__C.SUB_BATCH_SIZE: (accuracy_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = ques_ix_iter[accuracy_step * self.__C.SUB_BATCH_SIZE: (accuracy_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = ans_iter[accuracy_step * self.__C.SUB_BATCH_SIZE: (accuracy_step + 1) * self.__C.SUB_BATCH_SIZE]

                    pred = model(sub_frcn_feat_iter, sub_grid_feat_iter, sub_bbox_feat_iter, sub_ques_ix_iter)

                    loss_item = [pred, sub_ans_iter]
                    loss_nonlinear_list = self.__C.LOSS_FUNC_NONLINEAR[self.__C.LOSS_FUNC]
                    for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                        if loss_nonlinear in ['flat']:
                            loss_item[item_ix] = loss_item[item_ix].view(-1)
                        elif loss_nonlinear:
                            loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

                    loss = loss_fn(loss_item[0], loss_item[1])
                    if self.__C.LOSS_REDUCTION == 'mean':
                        loss /= self.__C.GRAD_ACCU_STEPS

                    loss.backward()
                    loss_tmp += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                if self.__C.VERBOSE:
                    if dataset_eval is not None:
                        mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                    else:
                        mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                    print("\r[INFO][Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e" %(
                        self.__C.VERSION,
                        self.__C.MODEL_USE,
                        self.DATASET,
                        epoch + 1,
                        step,
                        int(self.dataset.data_size / self.__C.BATCH_SIZE),
                        mode_str,
                        loss_tmp / self.__C.SUB_BATCH_SIZE,
                        optimizer._rate
                    ), end='        ')
                if self.__C.GRAD_NORM_CLIP  > 0:
                    torch.nn.clip_grad_norm_(model.parameters(), self.__C.GRAD_NORM_CLIP)


                for name in range(len(named_param)):
                    norm_v = torch.norm(named_param[name][1].grad).cpu().data.numpy() if named_param[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                optimizer.step()

            time_end = time.time()
            elapse_time = time_end - time_start
            print("[INFO] Finished in {}s".format(int(elapse_time)))
            epoch_finish = epoch + 1


            state = {
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.optimizer.state_dict(),
                'lr_base' : optimizer.lr_base,
                'epoch' : epoch_finish
            }

            torch.save(state, self.__C.SKPT_PATH + '/ckpt_' + self.__C.VERSION + '/epoch' + str(epoch_finish) + '.pkl')

            log_file = open(self.__C.LOG_PATH + '/log_run_' + self.__C.VERSION + '.txt', 'a+')
            log_file.write(
                'Epoch' + str(epoch_finish) +
                ', Loss' + str(loss_sum / self.dataset.data_size) +
                ', Lr' + str(optimizer._rate) + '\n' +
                ', Elapsed time' + str(int(elapse_time)) +
                ', Speed(s/batch): ' + str(elapse_time / step) +
                '\n\n'
            )
            log_file.close()

            if dataset_eval is not None:
                self.test(dataset_eval, state_dict = model.state_dict(), validation = True)

            loss_sum = 0
            grad_norm = np.zeros(len(named_param))

    def test(self, dataset, state_dict = None, validation = False):

        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, ''CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + '/ckpt_' + self.__C.CKPT_VERSION + '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        if state_dict is None:
            print("[INFO]Loading ckpt from: {}".format(path))
            state_dict = torch.load(path)['state_dict']
            print("[INFO]Finished!!!!")



        ans_ix_list = []
        pred_list = []
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        model = Model(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )

        model.cuda()
        model.eval()

        model.load_state_dict(state_dict)

        dataloader = DataLoader(
            dataset,
            batch_size = self.__C.EVAL_BATCH_SIZE,
            shuffle = False,
            num_workers = self.__C.NUM_WORKERS,
            pin_memory = self.__C.PIN_MEM
        )

        for step, (frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter, ans_iter) in enumerate(dataloader):
            print("\r[INFO] Evaluation: [step %4d/%4d]" %(step, int(data_size / self.__C.EVAL_BATCH_SIZE)), end = '     ')

        frcn_feat_iter = frcn_feat_iter.cuda()
        grid_feat_iter = grid_feat_iter.cuda()
        bbox_feat_iter = bbox_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()

        pred = model(frcn_feat_iter, grid_feat_iter, bbox_feat_iter, ques_ix_iter)

        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis = 1)
        if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(pred_argmax, (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]), mode = 'constant', constant_values = -1)

        ans_ix_list.append(pred_argmax)

        if self.__C.TEST_SAVE_PRED:
            if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_np = np.pad(pred_np, ((0, self.__C.EVAL_BATCH_SIZE), (0, 0)), mode = 'constant', constant_values = -1)
            pred_list.append(pred_np)

        print("")
        
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        
        if validation:
            if self.__C.RUN_MODE not in ['train']:
                result_eval_file = self.__C.CACHE_PATH +  '/result_run_' + self.__C.CKPT_VERSION
            else:
                result_eval_file = self.__C.CACHE_PATH + '/result_run_' + self.__C.VERSION
                
        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = self.__C.CACHE_PATH + '/result_run_' + self.__C.CKPT_VERSION
            else:
                result_eval_file = self.__C.CACHE_PATH + '/result_run_' + self.__C.CKPT_VERSION + '_epoch' + str(self.__C.CKPT_EPOCH)
                
        if self.__C.CKPT_PATH is not None:
            ensemble_file = self.__C.PRED_PATH + '/result_run_' + self.__C.CKPT_VERSION + '.pkl'
        else:
            ensemble_file = self.__C.PRED_PATH + '/result_run_' + self.__C.CKPT_VERSION + '_epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        if self.__C.RUNMODE not in ['train']:
            log_file = self.__C.LOG_PATH + '/log_run_' + self.__C.CKPT_VERSION + '.txt'
        else:
            log_file = self.__C.LOG_PATH + '/log_run_' + self.VERSION + '.txt'
                        
        eval_model(dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, validation)
        
        
        
    def empty_log(self, version):
        print("[INFO] Initializing log file......")
        if os.path.exists(self.__C.LOG_PATH + '/log_run_' + version + '.txt'):
            os.remove(self.__C.LOG_PATH + '/log_run_' + version + '.txt')
        print('Finished')
        print("")