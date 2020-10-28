#!/bin/bash 
# echo "Downloading dataset"
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
# wget http://images.cocodataset.org/zips/train2014.zip
# wget http://images.cocodataset.org/zips/val2014.zip
# wget http://images.cocodataset.org/zips/test2015.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
# wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip
# echo "Finished!!!!!!!!!!"


echo "Install the vncorenlp python wrapper"

pip install vncorenlp

mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

echo "Finished install of vncorenlp"