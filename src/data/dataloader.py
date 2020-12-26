import torch
import numpy as np
import glob
import json
from datetime import datetime
import time
import os


class Dataloader(torch.utils.data.Dataset):

    def __init__(self, __C, _type = 'train'):
        self.__C = __C 
        self.answer_dict = ''
        self._type = _type
        self.img_feat_path_list = glob.glob(self.__C.IMG_FEAT_PATH[self._type] + '*.npz')
        self.ques_feat_path = "./dataset/text/extract_bert"
        
        self.ques_list = json.load(open(self.__C.QUESTION_PATH[self._type], 'r'))["questions"]
        if self._type in ["train", "val"]:
            self.ans_list = json.load(open(self.__C.ANSWER_PATH[self._type], 'r'))["annotations"]
        
        if self._type == 'train':
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print(datetime.now().strftime("%H:%M:%S"), "\t INFO: DATASET SIZE = ", self.data_size)

        if self.__C.PRELOAD:
            print(datetime.now().strftime("%H:%M:%S"), "\t INFO: Pre-loading Features...")
            time_start = time.time()
            self.iid_to_img_feat = self.img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print(datetime.now().strftime("%H:%M:%S"), "\t INFO: Finished loading Features in {}s".format(int(time_end - time_start)))
        else:
            self.iid_to_img_feat_path = self.img_feat_path_load(self.img_feat_path_list)

        self.qid_to_ques = self.get_question_id(self.ques_list)
        self.question_features = self.load_ques_feat(os.path.join(self.ques_feat_path, self.BERT_METHOD, ".pt"))
        print(datetime.now().strftime("%H:%M:%S"), "\t INFO: Num qusetions loaded = ", len(self.question_features))


        self.ans_to_ix, self.ix_to_ans = self.ans_stat()
        self.ans_size = self.ans_to_ix.__len__()
        print(datetime.now().strftime("%H:%M:%S"), "\t INFO: Answer vocab size = ", self.ans_size)
        print("Finised!!!")

    def __getitem__(self, idx):

  
        img_feat_iter = None
        ques_ix_iter = None
        ans_iter = None


        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            qid = str(ans["question_id"])
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = self.proc_img_feat(img_feat_x)
            ques_ix_iter = self.proc_ques(qid)
            ans_iter = self.proc_ans(ans)

        else:
            ques = self.ques_list[idx]
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = self.proc_img_feat(img_feat_x)

            ques_ix_iter = self.proc_ques(qid)

        """
            return: 
                img_feat_iter : 1 x 100 x 1024
                ques_ix_iter  : 1 x 20 x 768 or 1024
                ans_iter      : vocab_ans x 1
        """
        return torch.from_numpy(img_feat_iter), torch.from_numpy(ques_ix_iter), torch.from_numpy(ans_iter)
        
    def __len__(self):
        return self.data_size

    def load_ques_feat(self, feat_file):
        feat_ques = torch.load(feat_file, map_location=torch.device("cpu"))
        assert len(self.qid_to_ques) == len(feat_ques), "ERROR"
        return feat_ques

    def get_question_id(self, ques_list):
        qid_to_ques = {}
        for ques in ques_list:
            qid = str(ques["question_id"])
            qid_to_ques[qid] = ques
        return qid_to_ques

    
    def img_feat_load(self, path_list):
        iid_to_feat = {}
        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            img_feat = np.load(path)
            img_feat_x = img_feat['x'].transpose((1, 0))
            iid_to_feat[iid] = img_feat_x
            print(datetime.now().strftime("%H:%M:%S"), "\t INFO  Pre Loading : [{} | {}]".format(ix, path_list.__len__()), end="\t\t")

        return iid_to_feat

    
    def img_feat_path_load(self, path_list):
        iid_to_path = {}
        for path in path_list:
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            iid_to_path[iid] = path
        return iid_to_path

    def ans_stat(self):
        ans_to_ix, ix_to_ans = json.load(open(self.answer_dict, 'r'))

        return ans_to_ix, ix_to_ans

    def proc_ques(self, qid):

        feat_ques = np.zeros((self.__C.TEXT_FEAT_PAD_SIZE, self.__C.WORD_EMBED_SIZE))
        for i, (tok, vec) in enumerate(self.question_features[qid]["vector"].values()):
            if i >= self.TEXT_FEAT_PAD_SIZE:
                break
            else:
                feat_ques[i] += vec

        ##return matrix MAX_TOKEN x 768 or 1024
        return feat_ques, question_features[qid]["idx_token"]
    
    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans):
        ans_score = np.zeros(self.ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = self.preprocess_ans(ans_['answers'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        for ans_ in ans_prob_dict:
            if ans_ in self.ans_to_ix:
                ans_score[self.ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])

        return ans_score

    def preprocess_ans(self, str):
        return ''

    def proc_img_feat(self, img_feat):
        if img_feat.shape[0] > self.__C.IMG_FEAT_PAD_SIZE:
            img_feat = img_feat[:self.__C.IMG_FEAT_PAD_SIZE]

        img_feat = np.pad(
            img_feat,
            ((0, self.__C.IMG_FEAT_PAD_SIZE - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat

    

        


