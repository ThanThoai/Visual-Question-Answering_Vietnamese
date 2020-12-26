<!-- # Visual-Question-Answering_Vietnamese

## Dataset
Sử dụng bộ dataset của [VQA] (https://visualqa.org/)

Sử dụng câu lệnh để download bộ dữ liệu: 

    chmod +x download.sh
    ./download.sh
    
Chuyển dữ liệu từ tiếng anh sang Tiếng Viêt:

Cây thư mục data:

    ./data: 
        |--train:
        |   |----v2_Annotations_Train_mscoco.json
        |   |----v2_Questions_Train_mscoco.json
        |   |----v2_Complementary_Pairs_Train_mscoco.json 
        |
        |
        |--val:
        |   |----v2_Annotations_Val_mscoco.json
        |   |----v2_Questions_Val_mscoco.json
        |   |----v2_Complementary_Pairs_Val_mscoco.json
        |
        |
        |--test:
        |   |----v2_Questions_Test_mscoco.json
    
Chạy lệnh để dịch dữ liệu:

    python3 trans.py
        

## Model
Project đang hướng tới việc áp dụng 2 model là **MCAN** (https://arxiv.org/abs/1906.10770) và **Oscar** (https://arxiv.org/abs/2004.06165).

## Tiền xử lý dữ liệu.
* Hình ảnh: Sử dụng phương pháp **Bottom up attention** (https://arxiv.org/pdf/1707.07998.pdf).
* Text: Sử dụng các phương pháp như **Word2Vec**, **Fasttext**, ...
 -->

<!-- ### Comming soon .......... -->
<!-- 
### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT   -->
 
---

<div align="center">    
 
# Visual Question Answering for Vietnamese     

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->


<!--  
Conference   
-->   
</div>
 
## Description   

Comming soon.............

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ThanThoai/Visual-Question-Answering_Vietnamese

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```