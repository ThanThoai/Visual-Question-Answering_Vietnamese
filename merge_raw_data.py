import os
import json
import tqdm
import argparse
import copy

def get_data_en(path_en, path_idx):

    result = {
        "train" : {},
        "val" : {},
        "test" : {},
        "test-dev" : {}
    }
    list_path_en = os.listdir(path_en)
    # list_path_idx = os.listdir(path_idx)
    count_ques = 0
    for f in list_path_en:
        if "train" in f:
            key = "train"
        elif "val" in f:
            key = "val"
        elif "testdev" in f:
            key = "test-dev"
        else:
            key = "test"
        with open(os.path.join(path_en, f), mode = 'r') as rb:
            data = rb.readlines()
        count_ques += len(data)
        id_ = f.split("_")
        with open(os.path.join(path_idx, "_".join(i for i in id_[:-1] + "_idx_" + id_[-1])), "r") as rb:
            idx = rb.readlines()

        for i, i_ in enumerate(idx):
            if i_.replace("\n", "") not in result[key].keys():
                result[key][i_.replace("\n", "")] = []
            result[key][i_.replace("\n", "")].append(data[i].replace("\n", ""))
    for key in result.keys():
        for k in result[key].keys():
            count_ques -= len(result[key][k])
    assert count_ques == 0, "Error"
    return result


def get_list_text(path_en, path_vi, path_idx, type_ = "train"):

    data_en = []
    data_vi = []
    data_idx = []

    list_en = os.listdir(path_en)
    for en in list_en:
        if type_ in en.split("_"):
            with open(os.path.join(path_en, en), mode = 'r') as rb:
                data_en += [i.replace("\n", "") for i in rb.readlines()]

            i_ = en.split("_")
            idx = "_".join(i for i in i_[:-1]) +  "_idx_" + i_[-1]
            i_ = en.split(".")
            vi = i_[0] + ".en.vi." + i_[1]
            with open(os.path.join(path_idx, idx), mode='r') as rb:
                data_idx += [i.replace("\n", "") for i in rb.readlines()]

            with open(os.path.join(path_vi, vi), mode = "r") as rb:
                data_vi += [i.replace("\n", "") for i in rb.readlines()]  

    print(len(data_en))
    print(len(data_vi))    
    assert len(data_en) == len(data_idx) and len(data_idx) == len(data_vi), "ERROR" 


    return data_en, data_vi, data_idx


def read_json(path_json):

    return json.load(open(path_json, 'rb'))

def check_string(str_1, str_2):
    return str_1 == str_2


def main(path_en, path_vi, path_idx, path_json, type_ = "train"):

    data_en, data_vi, data_idx = get_list_text(path_en, path_vi, path_idx, type_)
    data_json = read_json(path_json)
    dict_question = {}
    new_idx = copy.deepcopy(data_idx)
    for i, i_ in enumerate(data_idx):
        if i_.replace("\n", "") not in dict_question.keys():
            dict_question[i_.replace("\n", "")] = {}
        dict_question[i_.replace("\n", "")][i] = data_en[i].replace("\n", "")
    count_check = 0
    print(list(dict_question.keys())[:10])
    print(len(data_json['questions']))
    for question in data_json["questions"]:
        image_id = str(question["image_id"])
        qid = str(question["question_id"])
        q = question["question"]
        if image_id in dict_question.keys():
            for key, value in dict_question[image_id].items():
                if check_string(q, value):
                    new_idx[int(key)] = qid
                    count_check += 1
                    break
    print(count_check)
    with open(f"new_vi_{type_}.txt", "w") as wr:
        for question in data_vi[:-1]:
            wr.write(question)
            wr.write("\n")
        wr.write(data_vi[-1])
    with open(f"new_idx_{type_}.txt", "w") as wr:
        for idx in new_idx[:-1]:
            wr.write(idx)
            wr.write("\n")
        wr.write(new_idx[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert")
    parser.add_argument("--path_en_vi", dest="path_en_vi",
                        type =str,
                        default = "",
                        required = True)
    
    parser.add_argument("--path_idx", dest="path_idx",
                        type = str,
                        default = "",
                        required = True)   

    parser.add_argument("--path_file", dest="path_file",
                        type = str,
                        default = "",
                        required = True)     

    # parser.add_argument("--path_en", dest="path_en",
    #                     type = str,
    #                     default = "",
    #                     required = True)

    parser.add_argument("--type", dest = "type_",
                        type = str,
                        default = "",
                        required = True)        


    args = parser.parse_args()
    print(args)

    # main(args.path_en, args.path_en_vi, args.path_idx, args.path_file, args.type_)
    

    with open(args.path_en_vi, "r", encoding = "utf-8") as rb:
        data = rb.readlines()

    with open(args.path_idx, "r", encoding = "utf-8") as rb:
        idx = rb.readlines()

    with open(args.path_file, "rb") as rb:
        js = json.load(rb)
    vi_dict = {}
    for i, d in enumerate(data):
        vi_dict[idx[i].replace("\n", "")] = d
    print(len(vi_dict))
    print(idx[:10])
    print(list(vi_dict.keys())[:10])
    print(len(js["questions"]))
    count = []
    list_miss = []
    for i, question in tqdm.tqdm(enumerate(js["questions"])):
        # print(type(question["question_id"]))
        # print(question["question_id"])
        if question["question_id"] - int(idx[i].replace("\n", "")) == 0:
            question["question"] = data[i].replace("\n", "")
            count.append(i)
        else:
            list_miss.append(i)
    print(len(count))
    print(len(list_miss))
    json.dump(js, open(f"{args.type_}_vi.json", "w"))

    
        
    



        





