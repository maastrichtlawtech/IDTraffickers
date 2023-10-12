"""
Python version: 3.9
Description: Trains a transformers based classifier to establish baselines for Authorship tasks on Backpage advertisements.
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

import pandas as pd
import numpy as np

import torch

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
from loadData import HTClassifierDataModule

sys.path.append('../architectures/')
from HTTransferClassifier import HTTransferClassifierModel

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains a transformers based classifier to establish baselines for Authorship tasks on Backpage advertisements.")
parser.add_argument('--model_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the model to be trained (can only be between distilbert-base-cased)")
parser.add_argument('--tokenizer_name_or_path', type=str, default="johngiorgi/declutr-small", help="Name of the tokenizer to be used (can only be between distilbert-base-cased)")
parser.add_argument('--logged_entry_name', type=str, default="transferClassifierRoberta-temp:0.05-seed:1111", help="Logged entry name visible on weights and biases")
parser.add_argument('--data_dir', type=str, default='../data/processed/TEXT', help="""Data directory""")
parser.add_argument('--pickled_save_dir', type=str, default=os.path.join(os.getcwd(), "../pickled"), help="""Directory for data module to be saved""")
parser.add_argument('--demography', type=str, default='merged', help="""Demography of data, can be only between north, south, east, west, central, merged, or all""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/text-classifier-baselines/transfer"), help="""Directory for models to be saved""")
parser.add_argument('--pretrainedLM_path', type=str, default="/workspace/persistent/human-trafficking/models/lang-model/seed:1111/declutr-small/final_model.pt", help="""path of pre-trained LM""")
parser.add_argument('--pooling_type', type=str, default="max", help="""Can be mean, max, or mean-max""")
parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--patience', type=int, default=5, help="Patience for Early Stopping")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup proportion")
parser.add_argument('--grad_steps', type=int, default=4, help="Gradient accumulating step")
parser.add_argument('--learning_rate', type=float, default=6e-4, help="learning rate")
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
parser.add_argument('--train_data_percentage', type=float, default=1.0, help="Percentage of training data to be used")
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon value for adam optimizer")
parser.add_argument('--min_delta_change', type=float, default=0.01, help="Minimum change in delta in validation loss for Early Stopping")
parser.add_argument('--temp', type=float, default=0.07, help="Tempertaure variable for the loss function")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
parser.add_argument('--find_lr', action='store_true', help='Uses Lightning Tuner to find the best learning rate')
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
os.environ["WANDB_START_METHOD"] = "thread"
seed_everything(args.seed)

# Creating directories
directory = os.path.join(args.save_dir, "seed:" + str(args.seed), args.demography, args.pooling_type, args.model_name_or_path.split("/")[-1])
Path(directory).mkdir(parents=True, exist_ok=True)

# %% Loading the data
dm = HTClassifierDataModule(args)
dm.setup()

args.num_classes = pd.read_csv(os.path.join(args.data_dir, args.demography + '.csv'), error_bad_lines=False, warn_bad_lines=False).VENDOR.nunique()

args.num_training_steps = len(dm.train_dataloader()) * args.nb_epochs
# Setting the warmup steps to 1/10th the size of training data
args.warmup_steps = int(len(dm.train_dataloader()) * 10/100)

# %% Loading the model
model = HTTransferClassifierModel(args)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_change, patience=args.patience, verbose=False, mode="min")
model_checkpoint = ModelCheckpoint(dirpath=directory, filename="{epoch}-{step}-{val_loss:2f}", save_last=True, save_top_k=3, monitor="val_loss",  
                                    mode="min", verbose=True)
wandb_logger = WandbLogger(save_dir=os.path.join(directory, "logs"), name=args.logged_entry_name, project=args.demography + "_classifier")

# %% Setting up the trainer
# Unfortunately the lr_finder functionality doesn't support DeepSpeedStrategy yet, therefore we will set up our trainer twice. Once to find the suitable 
# learning rate and secondly to train our model. 

if args.find_lr:
    trainer = L.Trainer(max_epochs=args.nb_epochs, accelerator="gpu", devices=1 if torch.cuda.is_available() else None, fast_dev_run=False, 
                        accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                        benchmark = True, # Fastens the training process
                        deterministic=True, # Ensures reproducibility 
                        limit_train_batches=1.00, # trains on 10% of the data,
                        check_val_every_n_epoch = 1, # run val loop every 1 training epochs
                        precision = 16, # Mixed Precision system
                        )

    lr_finder = Tuner(trainer).lr_find(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(), min_lr=1e-6, max_lr=1, 
                                        num_training=100, mode='exponential', early_stop_threshold=4.0)
    model.hparams.learning_rate = lr_finder.suggestion()
    print("Learning Rate:", model.hparams.learning_rate)

trainer = L.Trainer(max_epochs=args.nb_epochs, accelerator="gpu", devices=1 if torch.cuda.is_available() else None, fast_dev_run=False, 
                    accumulate_grad_batches = args.grad_steps, # To run the backward step after n batches, helps to increase the batch size
                    benchmark = True, # Fastens the training process
                    deterministic=True, # Ensures reproducibility 
                    limit_train_batches=args.train_data_percentage, # trains on 10% of the data,
                    check_val_every_n_epoch = 1, # run val loop every 1 training epochs
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
torch.save(model.state_dict(), os.path.join(directory, "final_model.model"))

# %% Testing model performance
print("Train data performance:")
trainer.test(model=model, dataloaders=dm.train_dataloader())
print("Test data performance:")
trainer.test(model=model, dataloaders=dm.test_dataloader())
print("Validation data performance:")
trainer.test(model=model, dataloaders=dm.val_dataloader())

