"""
Python version: 3.8.12
Description: Extract the phone numbers and use them to create class labels for the advertisements. 
"""

#%% 
# %% Importing libraries
import networkx 
from networkx.algorithms.components.connected import connected_components

from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd

# Importing custom library
from utilities import extract_phone_from_texts, create_evaluation_file_for_crf_cnn, to_graph

# %% Loading data
data = pd.read_csv("../data/all_data.csv", sep="\t", on_bad_lines='skip', low_memory=False)
text = data.body.to_list()

# %% Extracting phone numbers using regex operations
noisy_phones, clean_phones = extract_phone_from_texts(text)
noisy_phones = [phones if len(phones) > 0 else None for phones in noisy_phones]
# Assigning the extracted phones to the pandas dataframe
# Total number of phone numbers extracted : 463,679
data['PHONES'] = noisy_phones

# %% Extracting the remaining phone numbers through the trained CRF-CNN module
data_cnn = data[data['PHONES'].isna()]
data_cnn['post_id'] = data_cnn.index.to_list()
# Splitting data into sequences of 50 characters to feed into the pre-trained CRF-CNN model
create_evaluation_file_for_crf_cnn(data_cnn, "../data/crf_cnn_split_data.tsv")

# %% Number of networks in the clean phone numbers
# Getting all the phone numbers with non None values
all_phones = [phones for phones in clean_phones if phones!=None]
# Filtering out numbers that are smaller than 7 digits
all_phones = [[phone for phone in phones if len(phone) >= 7] for phones in all_phones]
# Removing all the empty entries that had numbers smaller than 7 digits
all_phones = [phones for phones in all_phones if len(phones) > 0]

graph = to_graph(all_phones)
communities = connected_components(graph)
# Total number of communities: 309,346
phone_communities = [list(i) for i in communities]

