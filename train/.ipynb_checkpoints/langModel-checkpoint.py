"""
Python version: 3.9
Description: Trains a Language model on the Backpage Advertisement
Inspiration: https://github.com/yang-zhang/lightning-language-modeling
"""

# %% Importing libraries
import os
import sys
import random
import time
import datetime
import argparse
from pathlib import Path

import numpy as np

import torch
import lightning.pytorch as pl

from pytorch_lightning.loggers import WandbLogger

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.plugins.precision import DeepSpeedPrecisionPlugin

# Custom library
sys.path.append('../process/')
from loadData import LMDataModule

sys.path.append('../architectures/')
from LM import LMModel

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a language model using the sentence transformers declutr-small model on the Backpage advertisements.")
parser.add_argument('--model_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the model to be trained (can only be between distilbert-base-cased)")
parser.add_argument('--tokenizer_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the tokenizer to be used (can only be between distilbert-base-cased)")
parser.add_argument('--logged_entry_name', type=str, default="johngiorgi/declutr-small-temp:0.05-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='../data/processed/TEXT', help="""Data directory""")
parser.add_argument('--pickled_save_dir', type=str, default=os.path.join(os.getcwd(), "../pickled"), help="""Directory for data module to be saved""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/lang-model"), help="""Directory for models to be saved""")
parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=200, help="Number of Epochs")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--patience', type=int, default=5, help="Patience for Early Stopping")
parser.add_argument('--mlm_probability', type=float, default=0.15, help="The ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--preprocessing_num_workers', type=int, default=20, help='Number of workers for preprocessing')
parser.add_argument('--dataloader_num_workers', type=int, default=20, help='Number of workers for dataloader')
parser.add_argument('--grad_steps', type=int, default=4, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=0.00001, help="learning rate")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--temp', type=float, default=0.07, help="Tempertaure variable for the loss function")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--line_by_line', action='store_true', default=False)
parser.add_argument('--pad_to_max_length', action='store_true', default=False)
parser.add_argument('--overwrite_cache', action='store_true', default=False)
args = parser.parse_args()

# Setting seed value for reproducibility    
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
seed_everything(args.seed)

# Creating directories
directory = os.path.join(args.save_dir, "seed:" + str(args.seed),   args.model_name_or_path.split("/")[-1])
Path(directory).mkdir(parents=True, exist_ok=True)

# %% Loading the data module
train_filename = os.path.join(args.data_dir, "train_data.txt")
test_filename = os.path.join(args.data_dir, "test_data.txt")
dm = LMDataModule(tokenizer_name_or_path=args.tokenizer_name_or_path, train_file=train_filename, 
                            validation_file=test_filename, line_by_line=args.line_by_line, 
                            pad_to_max_length=args.pad_to_max_length, 
                            preprocessing_num_workers=args.preprocessing_num_workers,
                            overwrite_cache=args.overwrite_cache, max_seq_length=args.max_seq_length, 
                            mlm_probability=args.mlm_probability, train_batch_size=args.batch_size, 
                            val_batch_size=args.batch_size, dataloader_num_workers=args.dataloader_num_workers)
dm.setup()

args.num_training_steps = len(dm.train_dataloader()) * args.nb_epochs
# Setting the warmup steps to 1/10th the size of training data
args.warmup_steps = int(len(dm.train_dataloader()) * 10/100)

# %% Loading the model
model = LMModel(model_name_or_path=args.model_name_or_path, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                nr_training_steps=args.num_training_steps, decay=args.weight_decay, adam_epsilon=args.adam_epsilon)

# %% Setting up the trainer
early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
model_checkpoint = ModelCheckpoint(dirpath=directory, filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="valid_loss",  
                                    mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project="LM")

trainer = L.Trainer(max_epochs=args.nb_epochs, accelerator="gpu", devices=1 if torch.cuda.is_available() else None, fast_dev_run=False, 
                    accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                    benchmark = True, # Fastens the training process
                    deterministic=True, # Ensures reproducibility 
                    limit_train_batches=args.train_data_percentage, # trains on 10% of the data,
                    check_val_every_n_epoch = 1, # run val loop every 1 training epoch
                    callbacks=[model_checkpoint, early_stop_callback], # Enables model checkpoint and early stopping
                    logger = wandb_logger,
                    strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True), # Enable CPU Offloading, and offload parameters to CPU
                    plugins=DeepSpeedPrecisionPlugin(precision='16-mixed') # Mixed Precision system
                    )

# %% Training model
start_time = time.time()
trainer.fit(model, dm)
print("Total training:", str(datetime.timedelta(seconds=time.time()-start_time)))
trainer.save_checkpoint(os.path.join(directory, "final_model.ckpt"))

