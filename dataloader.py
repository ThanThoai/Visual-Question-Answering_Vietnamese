import numpy as np
import glob
import json
import re
from torch.utils import data
import tqdm
from vncorenlp import VnCoreNLP
import datetime
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from typing import List, Dict
import torch

class Tokenizer:

    def __init__(self, path = './vncorenlp/VnCoreNLP-1.1.1.jar'):

        self.path = path
        self.rdrsegenter = VnCoreNLP(self.path, annotators="wseg", max_heap_size='-Xmx500m')

    def tokenizer(self, sentences, return_string = False):

        re_sentences = self.rdrsegenter.tokenize(sentences)
        if return_string:
            return " ".join([s for s in re_sentences[0]])
        return re_sentences

class BPE_BASE():
    bpe_codes = './PhoBERT_base_fairseq/bpe.codes'

class BPE_LARGE():
    bpe_codes = "./PhoBERT_large_fairseq/bpe.codes"

class Embedding:

    def __init__(self, dict_word_question, max_token, BERT_MODEL = "BERT_BASE"):
        self.method = BERT_MODEL
        self.dict_model = {
            "BERT_BASE" : {
                "NAME" : "./PhoBERT_base_fairseq",
                "PATH_CHECKPOINT_FILE" : "model.pt"
            },

            "BERT_LARGE" : {
                "NAME" : "./PhoBERT_large_fairseq",
                "PATH_CHECKPOINT_FILE" : "model.pt"
            }

        }
        self.dict_word_question = dict_word_question
        assert self.method in self.dict_model.keys(), "[ERROR] Method {} not supported!!!!".format(self.method)
        self.pho_bert = RobertaModel.from_pretrained(self.dict_model[self.method]["NAME"], self.dict_model[self.method]["PATH_CHECKPOINT_FILE"])
        self.pho_bert.eval()
        if self.method == "BERT_BASE":
            args = BPE_BASE()
        else:
            args = BPE_LARGE()
        self.pho_bert.bpe = fastBPE(args)

    def extract(self, question: str):
        result = []
        doc = self.pho_bert.extract_features_aligned_to_words(question)
        for tok in doc[1: len(doc) - 1]:
            result.append(tok.vector)
        return np.array(result)


class Dataloader(data.Dataset):

    def __init__(self, ques_path, ans_path, word_path, feats_path, grids_path, mode = "train"):

        super(Dataloader, self).__init__()
        self.mode = mode
        self.frcn_feat_path_list = glob.glob(feats_path + '/*.npz')
        self.grid_feat_path_list = glob.glob(grid_path + '/*.npz')

        self.ques_dict = json.load(open(ques_path, 'r'))
        self.data_size = len(self.ques_dict)
        self.dict_word = json.load(open(word_path, 'r'))
        self.ans_dict = json.load(open(ans_path, 'r'))
        self.size_ans = len(self.ans_dict)
        self.embedding = Embedding(self.dict_word, max_token = 14)
        print("========== Dataset size:", len(self.ques_dict))
        self.iid_to_frcn_feat_path = self.img_feat_path_load(self.frcn_feat_path_list)
        self.iid_to_grid_feat_path = self.img_feat_path_load(self.grid_feat_path_list)
        self.qid_list = list(self.ques_dict.keys())
        print("========= Finished to loaded dataset")

    def img_feat_path_load(self, path_list):
        iid_to_path = {}
        print("Loading img features")
        for path in tqdm.tqdm(path_list):
            iid = path.split("/")[-1].split('.')[0]
            iid_to_path[iid] = path
        return iid_to_path



    def process_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[: img_feat_pad_size]
        img_feat = np.pad(img_feat,
                          ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
                          mode = "constant",
                          constant_values = 0)
        return img_feat

    def process_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype = np.float32)
        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])
        return bbox_feat


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_iter = self.process_img_feat(frcn_feat['x'], img_feat_pad_size = 100)

        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat_iter = grid_feat['x']

        bbox_feat_iter = self.process_img_feat(img_feat = self.process_bbox_feat(
                            bbox = frcn_feat['bbox'],
                            img_shape= (frcn_feat['height'], frcn_feat['width'])
                        ),
                        img_feat_pad_size = 100
        )
        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter

    def load_ques_ans(self, idx):
        qid = self.qid_list[idx]
        iid = self.ques_dict[qid]['imageId']
        question = self.ques_dict[qid]['question']
        answers = self.ques_dict[qid]['answers']
        question_idx = self.ques_dict[qid]['question_idx']
        lang_feat = self.embedding.extract(question)
        ans = np.zeros((1, self.size_ans), dtype = np.float32)
        ans[self.ans_dict[answers]['id']] = 1.0
        return lang_feat, question_idx, ans, iid

    def __getitem__(self, idx):

        lang_feat, question_idx, ans, iid = self.load_ques_ans(idx)

        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)

        return \
            torch.from_numpy(frcn_feat_iter),\
            torch.from_numpy(grid_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(lang_feat),\
            torch.from_numpy(question_idx), \
            torch.from_numpy(ans)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        np.random.shuffle(list)
