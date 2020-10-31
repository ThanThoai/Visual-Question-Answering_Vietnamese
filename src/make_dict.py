import sys
sys.path.append('../')
from utils.preprocesstext import prep_ans
from core.text import Tokenizer
from core.cfg import PATH
import json
import re

path = PATH()
tokenizer = Tokenizer()

ques_dict_preread = {
    'train': json.load(open(path.RAW_PATH['TRAIN'], 'rb')),
    'val': json.load(open(path.RAW_PATH['VAL'], 'rb')),
    'test': json.load(open(path.RAW_PATH['TEST'], 'rb')),
}

stat_ques_dict = {
    **ques_dict_preread['train'],
    **ques_dict_preread['val'],
    **ques_dict_preread['test'],
}

stat_ans_dict = {
    **ques_dict_preread['train'],
    **ques_dict_preread['val'],
}


def tokenize(stat_ques_dict):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        'CLS': 2,
    }

    max_token = 0
    for qid in stat_ques_dict:
        ques = stat_ques_dict[qid]['question']
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        )
        
        words = tokenizer.tokenize(words, return_string = False)

        if len(words) > max_token:
            max_token = len(words)

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)

    return token_to_ix, max_token


def ans_stat(stat_ans_dict):
    ans_to_ix = {}
    ix_to_ans = {}

    for qid in stat_ans_dict:
        ans = stat_ans_dict[qid]['answer']
        ans = prep_ans(ans)

        if ans not in ans_to_ix:
            ix_to_ans[ans_to_ix.__len__()] = ans
            ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans


if __name__ == "__main__":
    json_file = 'Dataset'
    token_to_ix, max_token = tokenize(stat_ques_dict)
    ans_to_ix, ix_to_ans = ans_stat(stat_ans_dict)

    json.dump([ans_to_ix, ix_to_ans, token_to_ix, max_token], open(json_file))