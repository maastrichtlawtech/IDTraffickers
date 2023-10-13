# IDTraffickers: An Authorship Attribution Dataset to link and connect Potential Human-Trafficking Operations on Text Escort Advertisements [(EMNLP 2023)](https://arxiv.org/abs/2310.05484)

Human trafficking (HT) is a pervasive global issue affecting vulnerable individuals, violating their fundamental human rights. Investigations reveal that a significant number of HT cases are associated with online advertisements (ads), particularly in escort markets. Consequently, identifying and connecting HT vendors has become increasingly challenging for Law Enforcement Agencies (LEAs). To address this issue, we introduce IDTraffickers, an extensive dataset consisting of 87,595 text ads and 5,244 vendor labels to enable the verification and identification of potential HT vendors on online escort markets. To establish a benchmark for authorship identification, we train a DeCLUTR-small model, achieving a macro-F1 score of 0.8656 in a closed-set classification environment. Next, we leverage the style representations extracted from the trained classifier to conduct authorship verification, resulting in a mean r-precision score of 0.8852 in an open-set ranking environment. Finally, to encourage further research and ensure responsible data sharing, we plan to release IDTraffickers for the authorship attribution task to researchers under specific conditions, considering the sensitive nature of the data. We believe that the availability of our dataset and benchmarks will empower future researchers to utilize our findings, thereby facilitating the effective linkage of escort ads and the development of more robust approaches for identifying HT indicators.

![](https://github.com/vageeshSaxena/IDTraffickers-EMNLP2023/blob/main/Images/Screenshot%20from%202023-10-12%2022-18-06.png)

# [IDTraffickers Dataset](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/NZ7VLC)

The dataset utilized in the experiments comprises text sequences created by combining the titles and descriptions of Backpage escort markets collected from 41 states from December 2015 to April 2016. The experiments are performed by establishing the ground truth with individual advertisements, achieved through the connection of phone numbers present in the advertisements. 

<p align="center">
  <img src="https://github.com/vageeshSaxena/IDTraffickers-EMNLP2023/blob/main/Images/Screenshot%20from%202023-10-12%2022-18-53.png" width="450" height="300">
</p>

# Setup
This repository is tested on Python 3.8+ and [conda](https://docs.conda.io/projects/miniconda/en/latest/). First, you should install a virtual environment:
```
conda create -n ID python=3.9
```

To activate the conda environment, run:
```
conda activate ID
```

Then, you can install all dependencies:
```
pip install -r requirements.txt
```

Additionally, to perform the authorship verification task, please install the FAISS package as suggested [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)


# Experiments
### Authorship Identification: A classification task
This research establishes a benchmark by performing the vendor identification task using various classifiers built over transformers-based contextualized embeddings. Given a specific text, the objective of the classifier is to predict the vendor that posted the advertisement. As shown below, the classifier performs the best when initialized with the representations from the DeCLUTR-small architecture. 
<p align="center">
  <img src="https://github.com/vageeshSaxena/IDTraffickers-EMNLP2023/blob/main/Images/Screenshot%20from%202023-10-12%2022-19-17.png" width="450" height="500">
</p>

The training details of all the trained classifiers can be found below:
<p align="center">
  <img src="https://github.com/vageeshSaxena/IDTraffickers-EMNLP2023/blob/main/Images/Screenshot%20from%202023-10-12%2022-20-22.png" width="800" height="500">
</p>

To train the classifier, run:
```
python train/classifier.py --batch_size 32 --demography merged --model_name_or_path johngiorgi/declutr-small --tokenizer_name_or_path johngiorgi/declutr-small --seed 1111 --logged_entry_name declutr-small-seed:5000-bs:32  --learning_rate 0.0001
```

### Authorship Verification: A retrieval task
Based on the style representations from the trained classifier, we also establish an authorship verification benchmark through an open-setting text-similarity-based ranking task, where we compute the cosine similarity between the style representations to analyze the patterns in writing style and determine if they came from the same vendor. 

<p align="center">
  <img src="https://github.com/vageeshSaxena/IDTraffickers-EMNLP2023/blob/main/Images/Screenshot%20from%202023-10-12%2022-19-43.png" width="800" height="500">
</p>

To conduct the retrieval task, we first [extract the trained representations](https://github.com/vageeshSaxena/IDTraffickers/blob/main/analysis/extract_embeddings.ipynb) and then [retrieve similar advertisements using the FAISS framework](https://github.com/vageeshSaxena/IDTraffickers/blob/main/analysis/faiss.ipynb).
