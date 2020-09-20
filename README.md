# Visual-Question-Answering_Vietnamese

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
Project đang hướng tới việc áp dụng 2 model là MCAN (https://arxiv.org/abs/1906.10770) và Oscar (https://arxiv.org/abs/2004.06165).

## Tiền xử lý dữ liệu.
* Hình ảnh: Sử dụng phương pháp Bottom up attention (https://arxiv.org/pdf/1707.07998.pdf).
* Text: Sử dụng các phương pháp như Word2Vec, Fasttext, ...

