"""
version: 3.9
Description: Rather than processing and tokenizing the data over and over again, we decided to store it to save computation. In our training script we load the saved datamodule before the training
"""

# %% Importing Libraries
import os
import sys
import pickle
import argparse
import random
import time
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from hurry.filesize import size

import torch
from pytorch_lightning import seed_everything


from transformers import AutoConfig

import warnings
warnings.filterwarnings('ignore')

# Custom library
sys.path.append('../process/')
from loadData import HTDataModule
from utilities import calculate_size_of_dumped_file

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Training a transformers architecture in supervised contrastive learning fashion.")
parser.add_argument('--model_name', type=str, default="roberta-base", help="Can be bert-base-cased or roberta-base")
parser.add_argument('--data_dir', type=str, default='../data/structured/TEXT', help="""Data directory""")
parser.add_argument('--demography', type=str, default='north', help="""Demography of data, can be only between north, south, east, west, or central""")
parser.add_argument('--pickled_save_dir', type=str, default=os.path.join(os.getcwd(), "../pickled"), help="""Directory for data module to be saved""")
parser.add_argument('--batch_size', type=int, default=40, help="Batch Size")
parser.add_argument('--max_seq_length', type=int, default=512, help="Maximum sequence length")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--split_ratio', type=float, default=0.20, help="Split ratio between training and test data")
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
Path(args.pickled_save_dir).mkdir(parents=True, exist_ok=True)

"""
# Raising exception for out of scope architectures
if args.model_name == "bert-base-cased" or args.model_name == "roberta-base":
    pass
else:
    raise Exception("Pipeline only implmented for bert-base-cased or roberta-base architecture")
"""

# Setting up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# %% Create an instance of LightningDataModule
dm = HTDataModule(data_dir=args.data_dir, model_name_or_dir=args.model_name, train_batch_size=args.batch_size, demography=args.demography, 
                    seed=args.seed, max_seq_length=args.max_seq_length, split_ratio=args.split_ratio)
# Fitting the data module
dm.setup(stage = 'fit')

pickle_filename = os.path.join(args.pickled_save_dir, "dm_" + args.model_name.split("/")[-1] + "_" + args.demography + ".pkl")

with open(pickle_filename, 'wb') as pfile:
    pickle.dump(dm, pfile, pickle.HIGHEST_PROTOCOL)

print("Total time taken by the script:", str(datetime.timedelta(seconds=time.time()-start_time)))
print("Total memory occupied by the dm object:", size(calculate_size_of_dumped_file(dm)))
