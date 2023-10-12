#!/bin/bash
# python classifier.py --batch_size 32 --demography south --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --find_lr > ../results/distilbert-base-cased-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32  --find_lr > ../results/deberta-v3-small-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --find_lr > ../results/albert-small-v2-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --find_lr > ../results/distilroberta-base-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --find_lr > ../results/distilgpt2-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --find_lr > ../results/t5-small-seed:1111-bs:32-south.txt

# python classifier.py --batch_size 32 --demography central --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --find_lr > ../results/distilbert-base-cased-seed:1111-bs:32-central.txt
# python classifier.py --batch_size 32 --demography central --model_name_or_path microsoft/deberta-v3-small --tokenizer_name_or_path microsoft/deberta-v3-base --logged_entry_name deberta-v3-small-seed:1111-bs:32  --find_lr > ../results/deberta-v3-small-seed:1111-bs:32-central.txt
# python classifier.py --batch_size 32 --demography central --model_name_or_path nreimers/albert-small-v2 --tokenizer_name_or_path nreimers/albert-small-v2 --logged_entry_name albert-small-v2-seed:1111-bs:32  --find_lr > ../results/albert-small-v2-seed:1111-bs:32-central.txt
# python classifier.py --batch_size 32 --demography central --model_name_or_path distilroberta-base --tokenizer_name_or_path distilroberta-base --logged_entry_name distilroberta-base-seed:1111-bs:32  --find_lr > ../results/distilroberta-base-seed:1111-bs:32-central.txt
# python classifier.py --batch_size 32 --demography central --model_name_or_path distilgpt2 --tokenizer_name_or_path distilgpt2 --logged_entry_name distilgpt2-seed:1111-bs:32  --find_lr > ../results/distilgpt2-seed:1111-bs:32-central.txt
# python classifier.py --batch_size 32 --demography central --model_name_or_path t5-small --tokenizer_name_or_path t5-small --logged_entry_name t5-small-seed:1111-bs:32  --find_lr > ../results/t5-small-seed:1111-bs:32-central.txt



# python classifier.py --batch_size 32 --demography central --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs:32  --learning_rate 0.00001 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-central.txt

# python classifier.py --batch_size 32 --demography north --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --logged_entry_name declutr-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/declutr-small-seed:1111-bs:32-north.txt

# python classifier.py --batch_size 32 --demography east --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --logged_entry_name declutr-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/declutr-small-seed:1111-bs:32-east.txt

# python classifier.py --batch_size 64 --demography south --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --logged_entry_name declutr-small-seed:1111-bs:64  --learning_rate 0.00001 > ../results/declutr-small-seed:1111-bs:64-south.txt
# python classifier.py --batch_size 32 --demography all --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --logged_entry_name declutr-small-seed:1111-bs:32  --learning_rate 0.00001 > ../results/declutr-small-seed:1111-bs:32-all.txt
# python classifier.py --batch_size 32 --demography all --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-all.txt
# python classifier.py --batch_size 32 --demography all --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-all.txt
# python classifier.py --batch_size 32 --demography all --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 --tokenizer_name_or_path sentence-transformers/all-MiniLM-L6-v2 --logged_entry_name all-MiniLM-L6-v2-seed:1111-bs:32  --learning_rate 0.00001 > ../results/all-MiniLM-L6-v2-seed:1111-bs:32-all.txt

# python classifier.py --batch_size 32 --demography all --model_name_or_path distilbert-base-cased --tokenizer_name_or_path distilbert-base-cased --logged_entry_name distilbert-base-cased-seed:1111-bs:32  --learning_rate 0.00001 > ../results/distilbert-base-cased-seed:1111-bs:32-all.txt


# python classifier.py --batch_size 32 --demography north --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-north.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-south.txt
# python classifier.py --batch_size 32 --demography south --model_name_or_path AnnaWegmann/Style-Embedding --tokenizer_name_or_path AnnaWegmann/Style-Embedding --logged_entry_name styleEmbedding-seed:1111-bs:32  --learning_rate 0.00001 > ../results/styleEmbedding-seed:1111-bs:32-south.txt