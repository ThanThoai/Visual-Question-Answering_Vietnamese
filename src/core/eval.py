from collections import defaultdict
from tqdm import tqdm
import os
import glob
import json 
import numpy as np
import pickle 

class Eval:
    
    def __init__(self, __C, result_eval_file, ques_file_path, choice_path = None, EVAL_CONSISTENCY = False):
        
        print("[INFO] Loading question......")
        questions = self.load_file(ques_file_path)
        
        if choice_path is not None:
            print("[INFO] Loading choices.....")
            choices = self.load_file(choice_path)
            
        print("[INFO] Loading predictions.....")
        self.predictions = self.load_file(result_eval_file)
        self.predictions = {
            p['question'] : p['prediction'] for p in self.predictions
        }
        
        for qid in questions:
            if (qid not in self.predictions) and (EVAL_CONSISTENCY or questions[qid]['isBalanced']):
                print("[INFO] No prediction for question {}.Please add prediction for all questions".format(qid))
                raise Exception("[ERROR] Missing predictions")
            
        self.scores = {
            "accuracy": [],
            "binary": [],
            "open" : [],
            "validity" : [],
            "pausibility" : [],
            "consistency" : [],
            "accuracyPerStructuralType" : defaultdict(list),
            "accuracyPerSemanticType" : defaultdict(list),
            "accuracyPerLength" : defaultdict(list),
            "accuracyPerSteps" : defaultdict(list),
            "grounding" : []
        }
        
        self.dist = {
            "gold" : defaultdict(lambda: defaultdict(int)),
            "predicted" : defaultdict(lambda: defaultdict(int))
        }
        
        for qid, question in tqdm(questions.items()):
            gold = question["answer"]
            predicted = self.prediction[qid]
            
            self.correct = (predicted == gold)
            score = self.to_score(self.correct)
            
            words_num = self.get_word_sum(question)
            steps_num = self.get_step_num(question)
            
            if question["isBalanced"]:
                self.scores["accuracy"].append(score)
                self.scores["accuracyPerLength"][words_num].append(score)
                self.scores["accuracyPerStep"][steps_num].append(score)
                self.scores["accuracyPerStructuralType"][question['types']['structural']].append(score)
                self.scores["accuracyPerSematicType"][question["types"]["sematic"]].append(score)
                answer_type = "open" if question['types']['structural'] == 'query' else 'binary'
                self.scores[answer_type].append(score)
                
                if choice_path is not None:
                    valid = self.belongs(predicted, choices[qid]['valid'], question)
                    self.scores["validity"].append(self.to_score(valid))
                    
                    plausible = self.belongs(predicted, choices[qid]['plausible'], question) 
                    self.scores["plausibility"].append(self.to_score(plausible))
                    
                global_group = question['groups']['global']
                if global_group is not None:
                    self.dist['gold'][global_group][gold] += 1
                    self.dist['predicted'][global_group][predicted] += 1
                
                if EVAL_CONSISTENCY:
                    self.upadte_consistency(qid, question, questions)
                
        self.scores['distribution'] = self.chi_square(self.dist['gold'], self.dist['predicted']) / 100
        
        metrics = [
            "binary",
            "open",
            "accuracy",
            "consistency",
            "validity",
            "plausibility",
            "grounding",
            "distribution"
        ]
        
        detaled_metrics = [
            ("accuracyPerStructuralType", "Accuracy / structural type"),
            ("accuracyPerSemanticType", "Accuracy / semantic type"),
            ("accuracyPerSteps", "Accuracy / steps number"),
            ("accuracyPerLength", "Accuracy / words number")
        ]              
        
        sub_metrics = {
            "attr": "attribute",
            "cat": "category",
            "global": "scene",
            "obj": "object",
            "rel": "relation"
        }
        
        for k in metrics:
            if isinstance(self.scores[k], list):
                self.scores[k] = self.avg(self.scores[k]) * 100
        
        for k, _ in detaled_metrics:
            for t in self.scores[k]:
                self.scores[k][t] = self.avg(self.scores[k][t]) * 100, len(self.scores[k][t])
                
        self.result_string = []
        self.detail_result_string = []
        
        for m in metrics:
            if m == 'grounding':
                continue
            if m == "consistency" and not EVAL_CONSISTENCY:
                continue
            if m == "validity" and choice_path is None:
                continue
            if m == "plausibility" and choice_path is None:
                continue
            
            self.result_string.append("{title}: {score:.2f}{suffix}".format(title= m.capitalize(), score= self.score[m],
                                                                    suffix= " (lower if better" if m == "distribution" else "%"))
            
        for m, m_print_name in detaled_metrics:
            self.detail_result_string.append("{}:".format(m_print_name))
            for t in sorted(list(self.scores[m].keys())):
                t_name = t
                if isinstance(self.scores[k], list):
                    t_name = sub_metrics.get(t, t).capitalize()
                self.detail_result_string.append(" {title}: {score:.2f}{suffix} ({amout} question)".format(title=t_name, score = self.scores[m][t][0], suffix = "%", amount = self.scores[m][t][1])) 
                       
    def get_str_result(self):
        return self.result_string, self.detail_result_string
    
    def load_file(self, name):
        if os.path.isfile(name):
            with open(name) as file:
                data = json.load(file)
        elif os.path.isdir(name.split(".")[0]):
            data = {}
            chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
            for chunk in chunks:
                with open(chunk) as file:
                    data.update(json.load(file))
        else:
            raise Exception("[ERROR] Can't find {}".format(name))
        
    def to_score(self, b):
        return float(1 if b else 0)
    
    def avg(self, l):
        if len(l) == 0:
            return 0
        return float(sum(l)) / len(l)
    
    def warg(self, l , w):
        if sum(w) == 0:
            return 0
        else:
            return float(sum(l[i] * w[i] for i in range(len(l)))) / len(l)
        
    def get_word_sum(self, question):
        return len(question["question"].split())
    
    def get_step_num(self, question):
        return len([c for c in question['sematic'] if not (any([o in "{}: {}".format(c["operation"], c["argument"]) for o in ["exits", "query: name", "choose name"]]))])
    
    def belongs(self, element, group, question):
        if "Common" in question["type"]["detailed"]:
            group = ["color", "material", "shape"]
        return element in group
    
    def upadte_consistency(self, question_id, question, questions):
        
        inferred_questrions = [eid for eid in question["entailed"] if eid != question_id]
        
        if self.correct and len(inferred_questrions) > 0:
            consistency_scores = []
            for eid in inferred_questrions:
                gold = questions[eid]["answer"]
                predicted = self.predictions[eid]
                score = self.to_score(predicted == gold)
                consistency_scores.append(score)
                
            self.scores["consistency"].append(self.avg(consistency_scores))
            
        
    def chi_square(self, gold_dist, predicted_dist):
        
        sum_score, sum_overall = 0, 0 
        for group in gold_dist:
            score, overall = 0, 0
            for ans in gold_dist[group]:
                e = gold_dist[group][ans]
                o = predicted_dist[group].get(ans, 0)
                score += ((float(o -e) ** 2) / e)
                overall += gold_dist[group][ans]
            sum_score += score * overall
            sum_overall += overall
            
        avg_score = float(sum_score) / sum_overall
        return avg_score
    
    

def eval_model(__C, dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, valid = False):
    
    result_eval_file = result_eval_file + '.json'
    qid_list = [qid for qid in dataset.qid_list]
    ans_size = dataset.ans_size
    
    result = [{
        "questionId" : qid_list[ix],
        "prediction" : dataset.ix_to_ans[str(ans_ix_list[ix])],
    } for ix in range(len(qid_list))]
    
    print("[INFO] Save the result to file: {}".format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))
    
    if __C.TEST_SAVE_PRED:
        print("[INFO] Save the prediction vector to file: {}".format(ensemble_file)) 
        
        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred' : pred_list[ix],
            'qid' : int(qid_list[ix])
        } for ix in range(qid_list.__len__())]
        
        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol = -1)
        
    if valid:
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        choice_path = None
        
        if __C.SPLIT['val'] + '_choices' in __C.RAW_PATH[__C.DATASET]:
            choice_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '_choices']
            
        eval_gqa = Eval(__C, result_eval_file, ques_file_path, choice_path, EVAL_CONSISTENCY = False)
        result_string, detail_result_string = eval_gqa.get_str_result()
        
        print("[INFO] Write to log file: {}".format(log_file))
        log = open(log_file, 'a+')
        for result_str in result_string:
            log.write(result_str + '\n')
            print(result_str)
            
        for detail_str in detail_result_string:
            log.write(detail_str + '\n')
        
        log.write('\n')
        log.close()
 
        