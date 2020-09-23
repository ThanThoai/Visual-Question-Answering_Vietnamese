
# from translate import Translator

# translator = Translator(to_lang="vi")
# print(translator.translate('this is a pen'))

import json 

with open('./question_train.json') as f:
    data = json.load(f)

# print(len(data['annotations']))

# print(len(data['questions']))
for i in range(10):
    print(data['questions'][i])
# type_quest = {}
# for i in range(len(data['annotations'])):
#     type_ = data['annotations'][i]['question_type']
#     if type_ not in type_quest.keys():
#         type_quest[type_] = 1
#     else:
#         type_quest[type_] += 1
        
# for key, value in type_quest.items():
#     print(key + ':' + str(value))
# # print(data['annotations'][1])

# # print(data['annotations'][2])

# # print(data['annotations'][3])



