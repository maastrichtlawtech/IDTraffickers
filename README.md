# [IDTraffickers: An Authorship Attribution Dataset to link and connect Potential Human-Trafficking Operations on Text Escort Advertisements](https://arxiv.org/abs/2310.05484)

Human trafficking (HT) is a pervasive global issue affecting vulnerable individuals, violating their fundamental human rights. Investigations reveal that a significant number of HT cases are associated with online advertisements (ads), particularly in escort markets. Consequently, identifying and connecting HT vendors has become increasingly challenging for Law Enforcement Agencies (LEAs). To address this issue, we introduce IDTraffickers, an extensive dataset consisting of 87,595 text ads and 5,244 vendor labels to enable the verification and identification of potential HT vendors on online escort markets. To establish a benchmark for authorship identification, we train a DeCLUTR-small model, achieving a macro-F1 score of 0.8656 in a closed-set classification environment. Next, we leverage the style representations extracted from the trained classifier to conduct authorship verification, resulting in a mean r-precision score of 0.8852 in an open-set ranking environment. Finally, to encourage further research and ensure responsible data sharing, we plan to release IDTraffickers for the authorship attribution task to researchers under specific conditions, considering the sensitive nature of the data. We believe that the availability of our dataset and benchmarks will empower future researchers to utilize our findings, thereby facilitating the effective linkage of escort ads and the development of more robust approaches for identifying HT indicators.

![](https://github.com/vageeshSaxena/IDTraffickers/blob/main/Images/Screenshot%20from%202023-10-12%2022-18-06.png)

# Dataset

The dataset utilized in the experiments comprises text sequences created by combining the titles and descriptions of Backpage escort markets collected from 41 states from December 2015 to April 2016. The experiments are performed by establishing the ground truth with individual advertisements, achieved through the connection of phone numbers present in the advertisements.

<p align="center">
  <img src="https://github.com/vageeshSaxena/IDTraffickers/blob/main/Images/Screenshot%20from%202023-10-12%2022-18-53.png" width="450" height="300">
</p>

# Setup
This repository is tested on Python 3.8+. First, you should install a virtual environment:
```
python3 -m venv .venv/HT
source .venv/HT/bin/activate
```

Then, you can install all dependencies:
```
pip install -r requirements.txt
```

# Experiments
### Authorship Identification: A classification task
