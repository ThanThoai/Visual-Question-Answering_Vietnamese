import os
import json
import tqdm
from utils.tokenizer import Tokenizer


def process_train(file_json, max_token):
    token = Tokenizer()
    dict_word = {
        "PAD" : 0,
        "UNK" : 1
    }
    result = {}
    data = json.load(open(file_json, 'r'))
    for key in tqdm.tqdm(data.keys()):
        question = token.tokenizer(data[key]['question'])[0]
        idx = np.zeros(max_token)
        for i, word in enumerate(question):
            if word not in dict_word:
                dict_word[word] = len(dict_word)
                idx[i] = len(dict_word)
        result[key] = {}
        result[key]['question'] = ' '.join([word for word in question])
        result[key]['answer'] = data[key]['answer']
        result[key]['imageId'] = data[key]['imageId']
        result[key]['question_idx'] = idx.tolist()

    json.dump(dict_word, open("dict_word.json", "w"))
    json.dump(result, open("train_vqa_vi.json", 'w'))


def get_text(file_json):
    data = json.load(open(file_json, 'r'))

    max_l = len(data)
    with open("idx_en.txt", "w") as iw:
        with open("word_en.txt", 'w') as wr:
            for w, i  in data.items():
                iw.write(str(i), "\n")
                wr.write(w + "\n")

def merge_text(file_en, file_vi, file_id):
    max_size = 1833
    result = {}
    data_en = open(file_en, 'r').readlines()[:max_size]
    data_vi = open(file_vi, 'r').readlines()[:max_size]
    data_id = open(file_id, "r").readlines()[:max_size]

    for i in range(len(data_en)):
        result[data_en[i]] = {}
        result[data_en[i]]['vi'] = data_vi[i]
        result[data_en[i]]['id'] = int(data_id[i])

    json.dump(result, open("answer_word.json", 'w'))
