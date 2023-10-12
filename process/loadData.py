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
from datasets import load_dataset
import lightning.pytorch as pl

from transformers import AutoTokenizer, DataCollatorForLanguageModeling

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
        train_dataset, val_dataset = random_split(train_dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42))
            
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

class LMDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name_or_path, train_file, validation_file, line_by_line, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"]
                                    if len(line) > 0 and not line.isspace()]
                return tokenizer(examples["text"], padding=padding, truncation=True, max_length=self.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=self.preprocessing_num_workers,
                                                remove_columns=[text_column_name], load_from_cache_file=not self.overwrite_cache,
                                                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,
            )

            if self.max_seq_length is None:
                self.max_seq_length = tokenizer.model_max_length
            else:
                if self.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to pHTClassifierDataModulereprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=self.mlm_probability)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=self.data_collator, 
                        num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.val_batch_size, collate_fn=self.data_collator, 
                            num_workers=self.dataloader_num_workers)