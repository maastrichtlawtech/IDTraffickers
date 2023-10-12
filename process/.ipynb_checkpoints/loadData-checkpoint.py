""" 
Python version: 3.9
Description: Contains helper classes and functions to load the data into the LightningDataModule.
"""

# %% Importing libraries
import os
import random

from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, TensorDataset
import lightning.pytorch as pl

from transformers import AutoTokenizer

# Define a custom dataset class for creating batches of sentence pairs
class HTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        # Constructor method that takes in a list of input pairs (data) and a tokenizer object
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        # Returns the length of the input data
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves the input pair at the given index in the data list
        input_pair = self.data[idx]
        # Tokenizes the input pair using the given tokenizer object, and returns the result as a PyTorch tensor
        encoded_pair = self.tokenizer(input_pair[0], input_pair[1], padding='max_length', truncation=True, max_length=self.max_seq_length, return_tensors='pt')
        return encoded_pair


class HTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, model_name_or_dir: str, demography: str = "north", seed: int = 1111,  max_seq_length: int = 512, 
                train_batch_size: int = 8, eval_batch_size: int = 1, split_ratio: float = 0.20, **kwargs):
        super().__init__()

        # Initialize the class attributes
        self.data_dir = data_dir
        self.demography = demography
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir, use_fast=True)
        self.nb_pos = 1
        self.nb_neg = train_batch_size - 1

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_and_split_data(self):
        # Load the data from a CSV file
        data_df = pd.read_csv(os.path.join(self.data_dir, self.demography + '.csv'), error_bad_lines=False, warn_bad_lines=False)

        # Initialize lists to hold the source texts, target texts, and labels
        source_texts, target_texts, label_list = ([] for i in range(3))

        # Get the unique vendors from the data
        all_vendors = data_df.VENDOR.unique()

        # Iterate through the data to find anchors, positives, and labels
        pbar = tqdm(total=len(all_vendors))
        for vendor in all_vendors:
            df = data_df[data_df["VENDOR"]==vendor]
            text_data = df.TEXT.to_list()
            
            # Find all possible pairs of source and target texts
            text_data = [(a, b) for idx, a in enumerate(text_data) for b in text_data[idx + 1:]]
            
            # Add the source texts, target texts, and labels to their respective lists
            source_texts.append([data[0] for data in text_data])
            target_texts.append([data[1] for data in text_data])
            label_list.append([vendor] * len(text_data))
            
            pbar.update(1)
        pbar.close()

        # Flatten the lists of source texts, target texts, and labels
        source_texts = [item for sublist in source_texts for item in sublist]
        target_texts = [item for sublist in target_texts for item in sublist]
        label_list = [item for sublist in label_list for item in sublist]

        data = list(zip(source_texts, target_texts, label_list))

        # Splitting the data
        train_data, test_data = train_test_split(data, test_size=self.split_ratio, random_state = self.seed, shuffle=True)
        train_data, val_data = train_test_split(train_data, test_size=0.05, random_state = self.seed, shuffle=True)
        return train_data, val_data, test_data 
    
    def get_positive_pairs(self, data, author):
        # Returns a generator expression that yields all positive pairs for the given author in the data
        return ((x[0], x[1]) for x in data if x[2] == author)

    def get_negative_pairs(self, data, author):
        author_dict = defaultdict(list)
        # Creates a dictionary with keys being all authors that are not the given author, 
        # and values being lists of negative pairs for each author
        for x in data:
            if x[2] != author:
                author_dict[x[2]].append((x[0], x[1]))
        return author_dict

    def get_batches(self, data):
        authors = set(x[2] for x in data)
        all_batches = []
        for author in authors:
            # Generates positive pairs for the current author
            positive_pairs = self.get_positive_pairs(data, author)
            # Generates negative pairs for all other authors
            negative_author_pairs = self.get_negative_pairs(data, author)
            for pos_pair in positive_pairs:
                current_batch = []
                current_batch.append(pos_pair)
                # Selects a random subset of authors to sample negative pairs from
                negative_authors = set(negative_author_pairs.keys())
                while len(current_batch) < self.nb_pos + self.nb_neg:
                    # Randomly selects an author to sample a negative pair from
                    random_negative_author = random.choice(list(negative_authors))
                    current_batch.append(random.choice(negative_author_pairs[random_negative_author]))
                    # Removes the selected author from the set of negative authors so it won't be selected again
                    negative_authors.remove(random_negative_author)
                all_batches.append(current_batch)
        return all_batches
    
    def setup(self, stage=None):
        # Load and split the data
        self.train_data, self.val_data, self.test_data = self.load_and_split_data()
        if stage == 'fit' or stage is None:
            self.train_batches = self.get_batches(self.train_data)
            self.val_batches = self.get_batches(self.val_data)

    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        dataset = HTDataset(self.train_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

    # Returning the pytorch-lightning default validation DataLoader
    def val_dataloader(self):
        dataset = HTDataset(self.val_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.eval_batch_size)

    # Returning the pytorch-lightning default test DataLoader
    def test_dataloader(self):
        test_batches = self.get_batches(self.test_data)
        dataset = HTDataset(test_batches, self.tokenizer, self.max_seq_length)
        return DataLoader(dataset, batch_size=self.eval_batch_size)

class HTClassifierDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Initialize the class attributes
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]

        # Handling the padding token in distilgpt2 by substituting it with eos_token_id
        if self.args.tokenizer_name_or_path == "distilgpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
    
    def setup(self, stage=None):
        # Load the dataset into a pandas dataframe.
        # Load the data from a CSV file
        data_df = pd.read_csv(os.path.join(self.args.data_dir, self.args.demography + '.csv'), error_bad_lines=False, warn_bad_lines=False)
        
        text = data_df.TEXT.values.tolist()
        vendors = data_df.VENDOR.values.tolist()
        
        # Tokenizing the data with padding and truncation
        encodings = self.tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', return_token_type_ids=False, truncation=True, 
                                   return_attention_mask=True, return_tensors='pt') 
                                   
        # Convert the lists into tensors.
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # Since the vendor IDs are not the current representations of the class labels, we remap these label IDs to avoid falling into out-of-bounds problem
        vendors_dict = {}
        i = 0
        for vendor in vendors:
            if vendor not in vendors_dict.keys():
                vendors_dict[vendor] = i
                i += 1
        vendors = [vendors_dict[vendor] for vendor in vendors]
        labels = torch.tensor(vendors)
        
        # Combine the inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_mask, labels)
                                   
        # Getting an 0.75-0.05-0.20 split for training-val-test dataset
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        train_dataset, self.val_dataset = random_split(train_dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42))
            
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        
    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset), batch_size=self.args.batch_size) 

    # Returning the pytorch-lightning default val DataLoader 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size) 
         
    # Returning the pytorch-lightning default test DataLoader 
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size) 

class HTClassifierForAllDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Initialize the class attributes
        if isinstance(args, tuple) and len(args) > 0: 
            self.args = args[0]

        # Handling the padding token in distilgpt2 by substituting it with eos_token_id
        if self.args.tokenizer_name_or_path == "distilgpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path, use_fast=True)
        
    def setup(self, stage=None):
        # Load the dataset into a pandas dataframe.
        # Load the data from a CSV file
        train_data_list, test_data_list, val_data_list = ([] for i in range(3))
        
        for demo in ["north", "east", "west", "south", "central"]:
            data_df = pd.read_csv(os.path.join(self.args.data_dir, demo + '.csv'), error_bad_lines=False, warn_bad_lines=False)

            text = data_df.TEXT.values.tolist()
            vendors = data_df.VENDOR.values.tolist()

            # Tokenizing the data with padding and truncation
            encodings = self.tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', return_token_type_ids=False, truncation=True, 
                                       return_attention_mask=True, return_tensors='pt') 

            # Convert the lists into tensors.
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            # Since the vendor IDs are not the current representations of the class labels, we remap these label IDs to avoid falling into out-of-bounds problem
            vendors_dict = {}
            i = 0
            
            for vendor in vendors:
                if vendor not in vendors_dict.keys():
                    vendors_dict[vendor] = i
                    i += 1
            vendors = [vendors_dict[vendor] for vendor in vendors]
            labels = torch.tensor(vendors)

            # Combine the inputs into a TensorDataset.
            dataset = TensorDataset(input_ids, attention_mask, labels)

            # Getting an 0.75-0.05-0.20 split for training-val-test dataset
            train_data, test_data = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
            train_data, val_data = random_split(train_data, [0.95, 0.05], generator=torch.Generator().manual_seed(42))
            
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            val_data_list.append(val_data)
            
        train_dataset = torch.utils.data.ConcatDataset(train_data_list)
        test_dataset = torch.utils.data.ConcatDataset(test_data_list)
        val_dataset = torch.utils.data.ConcatDataset(val_data_list)
            
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
            
    # Returning the pytorch-lightning default training DataLoader 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset), batch_size=self.args.batch_size) 

    # Returning the pytorch-lightning default val DataLoader 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size) 
         
    # Returning the pytorch-lightning default test DataLoader 
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size) 