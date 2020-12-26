import os
import json
import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert")
    parser.add_argument("--path_en_vi", dest="",
                        type =str,
                        required = True)
    
    parser.add_argument("--path_idx", dest="",
                        type = str,
                        required = True)   

    parser.add_argument("--path_file", dest="",
                        type = str,
                        required = True)             


    args = parser.parse_args()

    path_en_vi = args.path_en_vi
    path_idx   = args.path_idx
    file_path = args.path_file

    dict_file = {
        "train" : [],
        "val" : [],
        "test" : [],
        "test-dev" : []
    }

    list_file = os.listdir(path_en_vi)
    for file in list_file:
        if "train" in file:
            dict_file["train"].append(os.path.join(path_en_vi, file))
        elif "val" in file:
            dict_file["val"].append(os.path.join(path_en_vi, file))
        elif 'testdev' in file:
            dict_file["test-dev"].append(os.path.join(path_en_vi, file))
        else:
            dict_file["test"].append(os.path.join(path_en_vi, file))



    for t in dict_file.keys():
        data = {}
        for f in tqdm.tqdm(dict_file[t]):
            with open(f, 'r') as rb:
                d = rb.readlines()
            p_i = f.split("/")[-1].replace("en.vi", "").split("_")
            p_idx = p_i[0] + "_idx_" + p_i[1]
            with open(os.path.join(path_idx, p_idx), 'r') as rb:
                idx = rb.readlines()
            
            for i, _i in enumerate(idx):
                data[_i] = d[i]
        if t == 'train' or t == 'val':
            js = json.load(open(os.path.join(file_path, f"v2_OpenEnded_mscoco_{t}2014_questions.json"), 'rb'))
        else:
            js = json.load(open(os.path.join(file_path, f"v2_OpenEnded_mscoco_{t}2015_questions.json"), 'rb'))
        for j in tqdm.tqdm(js['question']):
            j["question"] = data[j['question_id']]

        json.dump(open(f"{t}_vi.json"), js)
        

