#!/bin/bash

# Training model on East data
# CUDA_VISIBLE_DEVICES=0 python contraLearn.py --model_name bert-base-cased --logged_entry_name bert-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6.9e-05-bs:32 > ../results/bert-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6.9e-05-bs:32.txt
# CUDA_VISIBLE_DEVICES=0 python contraLearn.py --model_name roberta-base --logged_entry_name roberta-base-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:0.002754-bs:32 --lr 0.002754 > ../results/roberta-base-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:0.002754-bs:32.txt
# CUDA_VISIBLE_DEVICES=0 python contraLearn.py --model_name roberta-base --logged_entry_name roberta-base-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6e-4-bs:32-warmup:100 > ../results/roberta-base-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6e-4-bs:32-warmup:100.txt
# CUDA_VISIBLE_DEVICES=0 python contraLearn.py --model_name bert-base-cased --logged_entry_name bert-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6.9e-05-bs:32 > ../results/bert-cased-temp:0.07-seed:1111-data:north-nb_epochs:40-lr:6.9e-05-bs:32.txt

# Generating dataloaders
# python process_dm.py --model_name bert-base-cased --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_bert-base-cased-east.txt
# python process_dm.py --model_name roberta-base --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_roberta-base-east.txt
# python process_dm.py --model_name distilbert-base-cased --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_distilbert-base-cased-east.txt
# python process_dm.py --model_name gpt2 --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_gp2-east.txt
# python process_dm.py --model_name microsoft/deberta-base --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_microsoft-deberta-base-east.txt
# python process_dm.py --model_name t5-small --demography east --batch_size 20 > /workspace/persistent/human-trafficking/pickled/results/dm_t5-small-east.txt

# Training model with batch size of 32 for train_data and 1 for eval_data
# CUDA_VISIBLE_DEVICES=0 python contraLearn.py --model_name bert-base-cased --demography east --logged_entry_name bert-cased-temp:0.07-seed:1111-data:east-lr:0.0002089-bs:32 --lr 0.0002089 > ../results/bert-cased-temp:0.07-seed:1111-data:east-lr:0.0002089-bs:32.txt

# Training classifiers
# python classifier.py --batch_size 32 --demography north --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --find_lr > ../results/distilbert-base-cased-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography north --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32  --find_lr > ../results/deberta-v3-small-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography north --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --find_lr > ../results/albert-small-v2-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography north --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --find_lr > ../results/distilroberta-base-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography north --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --find_lr > ../results/distilgpt2-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography north --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --find_lr > ../results/t5-small-seed:1111-bs:32-north.txt

# # python classifier.py --batch_size 32 --demography east --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --learning_rate 0.3801 > ../results/distilbert-base-cased-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography east --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --learning_rate 0.0005 > ../results/albert-small-v2-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography east --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32  --learning_rate 0.00087 > ../results/deberta-v3-small-seed:1111-bs:32-east.txt

# python classifier.py --batch_size 32 --demography east --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --find_lr > ../results/distilroberta-base-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography east --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --find_lr > ../results/distilgpt2-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography east --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/t5-small-seed:1111-bs:32-east.txt

# python classifier.py --batch_size 32 --demography west --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --find_lr > ../results/distilbert-base-cased-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32  --find_lr > ../results/deberta-v3-small-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --find_lr > ../results/albert-small-v2-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --find_lr > ../results/distilroberta-base-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --find_lr > ../results/distilgpt2-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --find_lr > ../results/t5-small-seed:1111-bs:32-west.txt



# python classifier.py --demography east --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --nb_epochs 5 --train_data_percentage 0.05 --batch_size 96

# python classifier.py --batch_size 32 --demography east --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32 > ../results/deberta-v3-small-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography east --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilbert-base-cased-seed:1111-bs:32-east.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilbert-base-cased-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --learning_rate 0.00001 > ../results/albert-small-v2-seed:1111-bs:32-west.txt
# python classifier.py --batch_size 32 --demography west --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32 --learning_rate 0.00001 > ../results/t5-small-seed:1111-bs:32-west.txt
# python transferClassifier.py --batch_size 32 --demography all --logged_entry_name transferRobertaClassifier-mean-lastlayer-seed:1111-bs:32 --pooling_type mean --learning_rate 0.00001 > ../results/transferRobertaClassifier-mean-lastlayer-seed:1111-bs:32.txt
# python transferClassifier.py --batch_size 32 --demography all --logged_entry_name transferRobertaClassifier-max-lastlayer-seed:1111-bs:32 --pooling_type max --learning_rate 0.00001 > ../results/transferRobertaClassifier-max-lastlayer-seed:1111-bs:32.txt
# python transferClassifier.py --batch_size 32 --demography all --logged_entry_name transferRobertaClassifier-mean-max-lastlayer-seed:1111-bs:32 --pooling_type mean-max --learning_rate 0.00001 > ../results/transferRobertaClassifier-mean-max-lastlayer-seed:1111-bs:32.txt
# python transferClassifier.py --demography north --logged_entry_name transferRobertaClassifier-mean-max-lastlayer-seed:1111-bs:32 --pooling_type mean-max --train_data_percentage 0.1 

# python classifier.py --batch_size 32 --demography merged --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilbert-base-cased-seed:1111-bs:32-merged.txt
python classifier.py --batch_size 32 --demography merged --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32 --learning_rate 0.00001 > ../results/deberta-v3-small-seed:1111-bs:32-merged.txt
# python classifier.py --batch_size 32 --demography merged --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --learning_rate 0.00001 > ../results/albert-small-v2-seed:1111-bs:32-merged.txt
python classifier.py --batch_size 32 --demography merged --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilroberta-base-seed:1111-bs:32-merged.txt
python classifier.py --batch_size 32 --demography merged --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilgpt2-seed:1111-bs:32-merged.txt
# python classifier.py --batch_size 32 --demography merged --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/t5-small-seed:1111-bs:32-merged.txt
python classifier.py --batch_size 32 --demography merged --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-merged.txt
# python classifier.py --batch_size 32 --demography merged --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs:32  --learning_rate 0.0001 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-merged.txt
python classifier.py --batch_size 32 --demography merged --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --logged_entry_name declutr-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/declutr-small-seed:1111-bs:32-merged.txt

python classifier.py --batch_size 32 --demography merged --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs-lr-0.001:32  --learning_rate 0.001 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-merged-lr-0.001.txt

python classifier.py --batch_size 32 --demography merged --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs-lr-0.0001:32  --learning_rate 0.0001 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-merged-lr-0.0001.txt

python transferClassifier.py --demography merged --logged_entry_name transferRobertaClassifier-mean-lastlayer-seed:1111-bs:32 --pooling_type merged --train_data_percentage --learning_rate 0.00001 > ../results/transferRobertaClassifier-mean-merged-lastlayer-seed:1111-bs:32.txt

# python classifier.py --batch_size 32 --demography north --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs:32  --lr 0.001 --nb_epochs 1 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-north-demo.txt

# python classifier.py --batch_size 32 --demography merged --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs:32  --learning_rate 0.00001