# Visual-Question-Answering_Vietnamese

## Dataset
Sử dụng bộ dataset của [VQA] (https://visualqa.org/)

Sử dụng câu lệnh để download bộ dữ liệu: 


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
        



