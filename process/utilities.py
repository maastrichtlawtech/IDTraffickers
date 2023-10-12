"""
Python version: 3.8.12
Description: Contains utilities and helper functions
"""

#%% Importing 
import sys
import gc
import re
import csv

from tqdm import tqdm


# Utilities and helper functions
def extract_phone_from_texts(texts):
    """
    param texts : input text (dtype: list)
    return : noisy phone numbers, and clean phone numbers
    """
    phone_noise_dict = {}

    phones_noisy = [list(set(re.findall(r'[\+\(]?[1-9][0-9 .\-\*\!\§\$\%\&\@\=\#\?\/\|\(\)]{8,}[0-9]', str(text)))) for text in texts]
    phones_clean = [split_extracted_phone_numbers_by_slash(phone) if len(phone) > 0 else None for phone in phones_noisy]
    # phones_clean = [[re.sub(r'[^\w]', ' ', number).replace(" ", "") if len(phone) > 0 else None for number in phone] for phone in phones_noisy if phone != None]
    phones_clean = [list(set(phone)) if phone != None else None for phone in phones_clean]

    return phones_noisy, phones_clean

def create_evaluation_file_for_crf_cnn(data, filename):
    data = data[['post_id', 'body']].set_index('post_id').to_dict()['body']

    id_list, text_list = ([] for i in range(2))
    pbar = tqdm(total=len(data))
    for id,text in data.items():
        text = [text[i-25:i+25] for i in range(25, len(text), 25)]
        for ad in text:
            id_list.append(id)
            text_list.append(ad)
        pbar.update(1)
    pbar.close()

    data = zip(id_list, text_list)
    with open(filename, 'w', newline='\n') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for id, val in data:
            tsv_output.writerow([id, val])

# Using recursion to merge all sublist having common phones and finding communities.
def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def split_extracted_phone_numbers_by_slash(noisy_phones):
    """
    Takes a list of noisy phone numbers, split them / if the number is > 10 and has a /
    in it, and finally returns a clean phone number.
    """
    inner_list = []
    for number in noisy_phones:
        if len(number) > 10:
            if "/" in number:
                numbers = number.split("/")
                for index, num in enumerate(numbers):
                    if len(num) < 10 and index != 0:
                        num = numbers[index-1][:len(numbers[index-1])-len(num)+1] + num
                        inner_list.append(num)
                    else:
                        inner_list.append(num)
            else:
                inner_list.append(number)
        else:
            inner_list.append(number)
    return inner_list

def calculate_size_of_dumped_file(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size