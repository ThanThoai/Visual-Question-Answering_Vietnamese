class Tokenizer:

    def __init__(self, path = './vncorenlp/VnCoreNLP-1.1.1.jar'):

        self.path = path
        self.rdrsegenter = VnCoreNLP(self.path, annotators="wseg", max_heap_size='-Xmx500m')

    def tokenizer(self, sentences, return_string = False):

        re_sentences = self.rdrsegenter.tokenize(sentences)
        if return_string:
            return " ".join([s for s in re_sentences[0]])
        return re_sentences
