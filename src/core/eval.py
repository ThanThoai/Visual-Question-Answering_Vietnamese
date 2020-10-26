from collections import defaultdict
from tqdm import tqdm
import os
import glob
import json as

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
                    valid = self.belogs(predicted, choices[qid]['valid'], question)
                    self.scores["validity"].append(self.to_score(valid))
                    
                    plausible = self.belogs(predicted, choices[qid]['plausible'], question) 
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
        