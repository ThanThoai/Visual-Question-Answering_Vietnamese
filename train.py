import datetime
from mcan.net import Net
import argparse
import torch
from utils.optim import Optim, get_optim, adjust_lr
from dataloader import Dataloader
import os
import json

def logs(s, verbose = True):
    if verbose:
        print(f"[{datetime.now().strftime(%m/%d/%Y, %H:%M:%S)}][INFO] {s}")
    with open("logs.txt", 'w') as wr:
        wr.write(s + '\n')

class CfgLoader():

    def __init__(self):

        self.DATA_ROOT = './data'

        self.FEAT_PATH = {
            "frcn" : os.path.join(self.DATA_ROOT, "feats", "frcn"),
            "grid" : os.path.join(self.DATA_ROOT, "feats", "grid")
        }

        self.QUESTION_PATH = ""
        self.ANSWER_WORD  = ""
        self.QUESTION_WORD = ""

        self.WORD_EMBED_SIZE = 768
        self.FEAT_SIZE = {
            "FFRCN_FEAT_SIZE" : (10, 2048),
            "GRID_FEAT_SIZE" : (49, 2048),
            "BBOX_FEAT_SIZE" : (100, 5)
        }

        self.BBOX_NORMALIZE = True
        self.NUM_WORKERS = 8
        self.PIN_MEM = True

        self.LR_BASE = 0.0001
        self.LR_DECAY_R = 0.2

        self.BATCH_SIZE = 64
        self.LR_DECAY_LIST = [8, 10]
        self.WARMUP_EPOCH = 2
        self.MAX_EPOCH = 11

        self.SEED = 42
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.SEED)
        random.seed(self.SEED)

        self.OPT_PARAMS = {
            "betas" : "(0.9, 0.98)",
            "eps" : "1e-9"
        }
        optim = getattr(torch.optim, "Adam")
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[3: optim.__init__.__code__.coargcount], optim.__init__.__defaults__[1:]))

        def all(iterable):
            for e in iterable:
                if not e:
                    return False
            return True

        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))

        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                exit(-1)
        self.OPT_PARAMS = {
            **default_params_dict,
            **self.OPT_PARAMS
        }

        self.LAYER = 6
        self.FF_SIZE = 2048
        self.HIDDEN_SIZE = 768
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.BBOXFEAT_EMB_SIZE = 2048
        self.USE_BBOX_FEAT = True
        self.USE_AUX_FEAT = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Training VQA for Vietnamese")
    parser.add_argument('--RESUME', dest='RESUME',
                        choices=['True', 'False'],
                        type=str)
    parser.add_argument('--PATH_LOAD_CHECKPOINT', dest='PATH_LOAD_CHECKPOINT',
                        type=str)
    parser.add_argument('--USE_GPU', dest='GPU',
                        choices=['True', 'False'],
                        type=str)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE',
                        type=int)
    parser.add_argument('--MAX_TOKEN', dest='MAX_TOKEN',
                        type=int)
    parser.add_argument('--EPOCH_SNAPSHOT', dest='EPOCH_SNAPSHOT',
                        type=int)
    parser.add_argument('--PATH_SAVE_CHECKPOINT', dest='PATH_SAVE_CHECKPOINT',
                        type=str)

    args = parser.parse_args()


    __C = CfgLoader()

    __C.BATCH_SIZE = args.BATCH_SIZE

    train_dataloader = Dataloader(ques_path = __C.QUESTION_PATH,
                                  ans_path  =__C.ANSWER_WORD,
                                  word_path = __C.QUESTION_WORD,
                                  feats_path = __C.FEAT_PATH['frcn'],
                                  grids_path = __C.FEAT_PATH['grid'])

    dataset = torch.utils.data.DataLoader(train_dataloader,
                                          batch_size = __C.BATCH_SIZE,
                                          shuffle = True,
                                          num_workers = __C.NUM_WORKERS,
                                          pin_memory = __C.PIN_MEM,
                                          drop_last = True)
    answer_size = train_dataloader.size_ans
    model = Net(__C, answer_size)
    devide = 'cpu'
    if args.GPU:
        devide = 'cuda'
    model.to(devide)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum').to(devide)
    start_epoch = 0
    data_size = train_dataloader.data_size
    optim = get_optim(__C, model, data_size)
    if args.RESUME:
        assert args.PATH_LOAD_CHECKPOINT != None, logs(s = "Not find pretrained model")
        logs("--- Resume training ---")
        logs(f"--- Loading model from {args.PATH_LOAD_CHECKPOINT}")
        try:
            pretrained_weight = torch.load(args.PATH_LOAD_CHECKPOINT)
            model.load_state_dict(pretrained_weight['static_dict'])
        except Exception as e:
            logs(e)

        start_epoch = pretrained_weight['epoch']
        optim = get_optim(__C, model, data_size = None, pretrained_weight['lr_base'])
        optim._step = int(data_size / args.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(pretrained_weight['optimizer'])


    named_params = list(model.named_parameters())
    grad_norm = np.zeros(len(named_params))
    for epoch in range(start_epoch, args.MAX_EPOCH):
        loss_sum = 0
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)
        for step, (frcn_feat_iter, grid_feat_iter, bbox_feat_iter, lang_feat, question_idx, ans) in enumerate(dataset):
            loss_tmp = 0
            pred = model(frcn_feat_iter, grid_feat_iter, bbox_feat_iter, lang_feat, question_idx)
            loss = loss_fn(pred, ans.view(-1))
            loss.backward()

            loss_tmp += loss.cpu().data.numpy()
            loss_sum += loss.cpu().data.numpy()
            if step % 100 == 0:
                logs("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                    epoch + 1,
                    step,
                    int(data_size / args.BATCH_SIZE),
                    loss_tmp / args.BATCH_SIZE,
                    optim._rate
                ))
        for name in range(len(named_params)):
            norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() if named_params[name][1].grad if not None else 0
            grad_norm[name] += norm_v
        optim.step()

        if epoch % args.EPOCH_SNAPSHOT == 0:
            state = {
                "state_dict" : model.state_dict(),
                "optimizer"  : optim.optimizer.state_dict(),
                "lr_base" : optim.lr_base,
                "epoch" : epoch + 1
            }
            torch.save(state, args.PATH_SAVE_CHECKPOINT)
        grad_norm = np.zeros(len(named_params))
