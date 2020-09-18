
# author : ThanThoai
# email  : thoai.bx163905@sis.hust.edu.vn

from translate import Translator
import json
import os
from tqdm import tqdm

class Translate_data:
    
    def __init__(self, path_dir  = 'data', dist_dir = 'data', list_keys = ['question_type', 'answer_type', 'multiple_choice_answer', 'answer', 'answer_confidence']):
        
        self.translator = Translator(to_lang = "vi")
        self.path_dir_train = path_dir + '/train'
        self.path_dir_test  = path_dir + '/test'
        self.path_dir_val   = path_dir + '/val'
        self.list_keys = list_keys
        if not os.path.isdir(os.path.join(dist_dir, 'data_vietnamese')):
            os.mkdir(os.path.join(dist_dir, 'data_vietnamese'))
        
        self.path_dist_dir_train = os.path.join(dist_dir, 'data_vietnamese', 'train')
        self.path_dist_dir_test  = os.path.join(dist_dir, 'data_vietnamese', 'test')
        self.path_dist_dir_val   = os.path.join(dist_dir, 'data_vietnamese', 'val')
        
        if not os.path.isdir(self.path_dist_dir_train):
            os.mkdir(self.path_dist_dir_train) 
        if not os.path.isdir(self.path_dist_dir_test):
            os.mkdir(self.path_dist_dir_test)           
        if not os.path.isdir(self.path_dist_dir_val):
            os.mkdir(self.path_dist_dir_val)
        
        
    def __dump_file(self, data, file_name):
        with open(file_name, "w") as f:
            json.dump(data, f)
        
        
    def __translate_train_file(self, file_name = 'v2_Questions_Train_mscoco.json'):
        
        def translate_train(path_file):
            
            with open(path_file, 'r') as f:
                data = json.load(f)
            
            results = {
                "info" : data['info'],
                "task_type" : data['task_type'],
                "data_subtype" : data['data_subtype'],
                "questions" : [],
                "license" : data['license'] 
            }
            
            for i in tqdm(range(len(data['questions']))):
                question = {
                    "question_id" : None,
                    "image_id"    : None,
                    "question"    : None
                }
                question['question_id'] = data['questions'][i]['question_id']
                question['image_id'] = data['questions'][i]['image_id']
                question['question'] = self.translator.translate(data['questions'][i]['question'])
                results['questions'].append(question)
            return results
        
  
        result_train = translate_train(os.path.join(self.path_dir_train, file_name))
        result_test  = translate_train(os.path.join(self.path_dir_test, file_name.replace('Train', 'Test')))
        result_val   = translate_train(os.path.join(self.path_dir_val, file_name.replace('Train', 'Val')))
        
        return result_train, result_test, result_val
    
    def __translate_annotation_file(self, file_name = "v2_Annotations_Train_mscoco.json", list_keys = ['question_type', 'answer_type', 'multiple_choice_answer', 'answer', 'answer_confidence']):
        
        def translate_annotation(path_file):
            
            with open(path_file, 'r') as f:
                data = json.load(f)
            
            results = {
                "info" : data["info"],
                "data_type" : data['data_type'],
                "data_subtype" : data["data_subtype"],
                "annotations" : [],
                "license" : data["license"]
            }
            
            for i in tqdm(range(len(data['annotations']))):
                
                annotation = {
                    "question_id" : data['annotations'][i]['question_id'],
                    "image_id" : data['annotations'][i]['image_id']
                    "question_type" : data['annotations'][i]['question_type']
                    "answer_type" : data['annotations'][i]['answer_type']
                    "answers" : [],
                    "multiple_choice_answer" : None,
                }
                
                if "multiple_choice_answer" in list_keys:
                    annotation['multiple_choice_answer'] = self.translator.translate(data['annotations'][i]['multiple_choice'])
                else:
                    annotation['multiple_choice_answer'] = data['annotations'][i]['multiple_choice_answer']
                
                for j in range(len(data['annotations'][i]['answers'])):
                    answer = {
                        "answer_id" : None,
                        "answer" : None,
                        "answer_confidence" :  None
                    }    
                    
                    for key in data['annotations'][i]['answers'][j].keys():
                        if key in list_keys:
                            answer[key] = self.translator.translate(data['annotations'][i]['answers'][j][key])
                        else:
                            answer[key] = data['annotations'][i]['answers'][j][key]
                    
                    annotation['answer'].append(answer)

                results['annotations'].append(annotation)     
            return results
        
        
        results_train = translate_annotation(os.path.join(self.path_dir_train, file_name))
        results_val   = translate_annotation(os.path.join(self.path_dir_val, file_name.replace("Train", "Val")))
        return results_train, results_val            
                            
        
    def translate(self, name_question = 'v2_Questions_Train_mscoco_vi.json', name_annotation = 'v2_Annotations_Train_mscoco_vi.json'):
        
        print("Process Question file")
        results_train, results_test, results_val = self.__translate_train_file()
        print("Dump to file")
        self.__dump_file(results_train, file_name = os.path.join(self.path_dist_dir_train, name_question))
        self.__dump_file(results_test,  file_name = os.path.join(self.path_dist_dir_test, name_question.replace('Train', 'Test')))
        self.__dump_file(results_val, file_name  = os.path.join(self.path_dist_dir_val, name_question.replace('Train', 'Val')))
        
        print("Process Annotation file")
        results_train, results_test = self.__translate_annotation_file(list_keys = self.list_keys)
        print("Dump to file")
        self.__dump_file(results_train, file_name = os.path.join(self.path_dist_dir_train, name_annotation))
        self.__dump_file(results_val, file_name = os.path.join(self.path_dist_dir_val, name_annotation.replace('Train', "Val")))
        
        print("Done")
            
        
        
if __name__ == "__main__":
    
    app = Translate_data()
    app.translate()
