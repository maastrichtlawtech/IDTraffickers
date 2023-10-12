"""
Python version: 3.9
Description: Trains a BERT-cased or RoBERTa architecture in semi-Supervised Contrastive Learning fashion
"""
# %% Importing Libraries
import os
import sys
import pickle
import argparse
import time
import datetime
import random
from pathlib import Path
from hurry.filesize import size

import pandas as pd
import numpy as np

import torch

from transformers import AutoConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

# Custom library
sys.path.append('../process/')
from loadData import HTDataModule
from utilities import calculate_size_of_dumped_file

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Training a BERT-cased or RoBERTa architecture in semi-supervised contrastive learning fashion.")
parser.add_argument('--model_name', type=str, default="roberta-base", help="Can be bert-base-cased or roberta-base")
parser.add_argument('--logged_entry_name', type=str, default="roberta-base-temp:0.05-seed:1111-data:north-nb_epochs:40", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='../data/structured/TEXT', help="""Data directory""")
parser.add_argument('--pickled_save_dir', type=str, default=os.path.join(os.getcwd(), "../pickled"), help="""Directory for data module to be saved""")
parser.add_argument('--demography', type=str, default='north', help="""Demography of data, can be only between north, south, east, west, or central""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/contra-learn/text-baselines/architectural"), help="""Directory for models to be saved""")
parser.add_argument('--pooler_type', type=str, default='cls', help="Can be cls, cls_before_pooler, avg, avg_top2, or avg_first_last")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--patience', type=int, default=3, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=500, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=4, help="Gradient accumulating step")
parser.add_argument('--lr', type=float, default=6e-4, help="learning rate")
parser.add_argument('--split_ratio', type=float, default=0.20, help="Split ratio between training and test data")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--eps', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.001, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--temp', type=float, default=0.07, help="Tempertaure variable for the loss function")
parser.add_argument('--decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--train', action='store_true', help='Initiates the training process')
parser.add_argument('--used_pickled_dataloader', action='store_true', help='Whether or not to use pre-tokenized data')
args = parser.parse_args()

# Setting seed value for reproducibility    
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
seed_everything(args.seed)

# Creating directories
Path(os.path.join(args.save_dir, args.model_name, args.logged_entry_name)).mkdir(parents=True, exist_ok=True)
# Raising exception for out of scope architectures
assert args.model_name in ["bert-base-cased", "roberta-base", "distilbert-base-cased", "gpt2", "microsoft/deberta-base", "t5-small"]

# Setting up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Loading the dataloader
if args.used_pickled_dataloader:
    pickle_filename = os.path.join(os.getcwd(), args.pickled_dir, "dm_" + args.model_name.split("/")[-1] + "_" + args.demography + ".pkl")
    with open(pickle_filename, 'rb') as inp_file:
        dm = pickle.load(inp_file)    
else:
    # %% Create an instance of LightningDataModule
    start_time = time.time()
    dm = HTDataModule(data_dir=args.data_dir, model_name_or_dir=args.model_name, train_batch_size=args.batch_size, demography=args.demography, 
                        seed=args.seed, max_seq_length=args.max_seq_length, split_ratio=args.split_ratio)
    # Fitting the data module
    dm.setup(stage = 'fit')

    pickle_filename = os.path.join(args.pickled_save_dir, "dm_" + args.model_name.split("/")[-1] + "_" + args.demography + ".pkl")
    with open(pickle_filename, 'wb') as pfile:
        pickle.dump(dm, pfile, pickle.HIGHEST_PROTOCOL)

    print("Total time taken to load the data:", str(datetime.timedelta(seconds=time.time()-start_time)))
    print("Total memory occupied by the dm object:", size(calculate_size_of_dumped_file(dm)))

# %% Loading the model
config = AutoConfig.from_pretrained(args.model_name)

if args.model_name=="bert-base-cased":
    # Loading the BERT layer for contrastive learning
    sys.path.append('../architectures/')
    from BERTLayer import BertForCL

    model = BertForCL(config=config, temp = args.temp, warmup_steps = args.warmup_steps, adam_epsilon = args.eps, weight_decay = args.decay, 
                        learning_rate = args.lr, pooler_type=args.pooler_type)

elif args.model_name=="roberta-base":
    # Loading the BERT layer for contrastive learning
    sys.path.append('../architectures/')
    from RoBERTaLayer import RoBertaForCL

    model = RoBertaForCL(config=config, temp = args.temp, warmup_steps = args.warmup_steps, adam_epsilon = args.eps, weight_decay = args.decay, 
                        learning_rate = args.lr, pooler_type=args.pooler_type)

# %% Setting the trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.save_dir, args.model_name, args.logged_entry_name), filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="val_loss",  mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(args.save_dir, args.model_name, args.logged_entry_name, "logs"), name=args.logged_entry_name, project='HTtransformers')

trainer = Trainer(max_epochs=args.nb_epochs, accelerator="gpu", devices=1 if torch.cuda.is_available() else None,  
                    accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                    benchmark = True, # Fastens the training process
                    limit_train_batches= args.train_data_percentage, # trains on train_data_percentage% of the data,
                    log_every_n_steps=1000, # logs after every 1000 steps
                    callbacks=[early_stop_callback, model_checkpoint], # Enabling early stopping,and saving the model checkpoints
                    strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True), # Enable CPU Offloading, and offload parameters to CPU
                    precision=16, # Precision system
                    logger=wandb_logger,  # Adding the logger
                )

trainer.fit(model, datamodule=dm)
trainer.save_checkpoint(os.path.join(args.save_dir, args.model_name, args.logged_entry_name, 'EarlyStoppingAdam-' + args.logged_entry_name + ".pth"))