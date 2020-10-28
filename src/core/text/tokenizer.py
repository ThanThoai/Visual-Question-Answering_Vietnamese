from vncorenlp import VnCoreNLP
import torch 
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
import numpy as np

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

    def __init__(self, __C, method):

        assert method in ["Bert", "Fasttext", "Word2Vec"], "[ERROR] Method embeding not support!!!"
        self __C = __C
        self.method = method
        self.tokenizer = Tokenizer()
        if self.method == "Bert":
            # self.pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
            # self.tokenizer_bert = AutoTokenizer.from_pretrained("vinai/phobert-base")

            self.pho_bert = RobertaModel(self.__C.EMBEDDING_MODEL['BERT']['NAME'], self.__C.EMBEDDING_MODEL['BERT']["PATH_CHECKPOINT_FILE"])
            self.pho_bert.eval()
            args = BPE()
            self.pho_bert.bpe = fastBPE(args)

        elif self.method == "":
            pass 
            #TODO 
        else:
            pass 
            #TODO


    def embeding(self, sentences):

        sentences = self.tokenizer(sentences, return_string = True)
        if self.method == "Bert":
            return self.embeding_bert(sentences)
        elif self.method == "Fasttext":
            return self.embeding_fasttext(sentences)
        else:
            return self.embeding_word2vec(sentences)


    def embedding_bert(self, sentences, method = 'avg'):
        assert method in ['avg', 'sum'], "[ERROR] method {} not supported".format(method)

        doc = self.pho_bert.extract_features_aligned_to_words(sentences)
        if  method == 'avg':
            features = np.mean(doc, axis = 1)
        else:
            features = np.sum(doc, axis = 1)
        return features

    def embedding_fasttext(self, sentences):
        pass 
        #TODO

    def embedding_word2vec(self, sentences):
        pass 
        #TODO

    

            
