from vncorenlp import VnCoreNLP
import torch 
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
import numpy as np
import json

class Tokenizer:

    def __init__(self, path = '/Absolute-path-to/vncorenlp/VnCoreNLP-1.1.1.jar'):

        self.path = path 
        self.rdrsegenter = VnCoreNLP(self.path, annotators="wseg", max_heap_size='-Xmx500m')

    def tokenizer(self, sentences, return_string = False):
        
        re_sentences = self.rdrsegmenter.tokenizer(sentences)
        if return_string:
            return " ".join([s for s in re_sentences])
        return re_sentences


class BPE():
    bpe_codes = 'PhoBERT_base_faiers/bpe.codes'

class Embedding:
    
    def __init__(self, __C, token_to_ix):
        
        self.__C = __C
        self.token_to_ix = token_to_ix
        self.method = self.__C["METHOD"]
        assert self.ethod in ["BERT_SMALL", "BERT_LARGE"], "[ERROR] Method {} not supported!!!!".format(self.method)
        self.pho_bert = RobertaModel(self.__C.EMBEDDING_MODEL[self.method]['NAME'], self.__C.EMBEDDING_MODEL[self.method]["PATH_CHECKPOINT_FILE"])
        self.pho_bert.eval()
        args = BPE()
        self.pho_bert.bpe = fastBPE(args)

    def run(self):
        pretrained_emb = []
        for word in self.token_to_ix:
            vector = self.pho_bert.extract_features_aligned_to_words(word)
            pretrained_emb.append(vector)
            
        return np.array(pretrained_emb)

            
