# from core.data.ans_punct import prep_ans
import numpy as np
import random
import re
import json
from datetime import datetime
import os
from vncorenlp import VnCoreNLP
import torch 
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
import tqdm
from typing import List, Dict
import argparse



class Tokenizer:

    def __init__(self, path = './vncorenlp/VnCoreNLP-1.1.1.jar'):

        self.path = path 
        self.rdrsegenter = VnCoreNLP(self.path, annotators="wseg", max_heap_size='-Xmx500m')

    def tokenizer(self, sentences, return_string = False):
        
        re_sentences = self.rdrsegenter.tokenize(sentences)
        if return_string:
            return " ".join([s for s in re_sentences])
        return re_sentences

class BPE_BASE():
    bpe_codes = './PhoBERT_base_fairseq/bpe.codes'

class BPE_LARGE():
    bpe_codes = "./PhoBERT_large_fairseq/bpe.codes"

class Embedding:
    
    def __init__(self, BERT_MODEL, write_file = True):
        


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
        assert self.method in self.dict_model.keys(), "[ERROR] Method {} not supported!!!!".format(self.method)
        self.pho_bert = RobertaModel(self.dict_model[self.method]["NAME"], self.dict_model[self.method]["PATH_CHECKPOINT_FILE"])
        self.tokenize = Tokenizer()
        self.pho_bert.eval()
        if self.method == "BERT_BASE":
            args = BPE_BASE()
        else:
            args = BPE_LARGE()
        self.pho_bert.bpe = fastBPE(args)
        self.embeding = {}
        self.write_file = write_file

    def run(self, data : List[Dict]) -> None:
        print(datetime.now().strftime("%H:%M:%S"), f"\t INFO: Extract Features using BERT: {self.method}")
        for d in tqdm.tqdm(data):
            key = d['question_id']
            assert key not in self.embeding.keys(), "ERROR"
            question = self.tokenize.tokenizer(d["question"], return_string = True)
            token_idx = self.pho_bert.encode(question)
            doc = self.pho_bert.extract_features_aligned_to_words(question)
            self.embeding[key] = {
                "vector" : {},
                "idx_token" : token_idx[1:-1]
            }
            for tok in doc[1: -1]:
                self.embeding[key]["vector"][str(tok)] = tok.vector
        
        if self.write_file:
            path = "./dataset/text/extract_bert"
            print(datetime.now().strftime("%H:%M:%S"), f"\t INFO: Saving extract features to file")
            torch.save(self.embeding, os.path.join(path, self.method + '.pt'))


    def extract(self, question: str) -> Dict:
        result = {}
        doc = self.pho_bert.extract_features_aligned_to_words(self.tokenize.tokenizer(question))
        for tok in doc[1: len(doc) - 1]:
            result[tok] = tok.vector
        return result




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Features Text")
    parser.add_argument("--MODEL", dest="RUN MODEL BERT",
                        choices= ["BASE", "LARGE"],
                        type =str,
                        required = True)
    
    parser.add_argument("--DATA", dest="DATA",
                        type = str,
                        required = True)     


    args = parser.parse_args()
    
    App = Embedding(args.MODEL)
    data = json.load(open(args.DATA, "rb"))
    App.run(data["question"])