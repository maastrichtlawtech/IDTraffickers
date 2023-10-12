# IDTraffickers: An Authorship Attribution Dataset to link and connect Potential Human-Trafficking Operations on Text Escort Advertisements

Human trafficking (HT) is a pervasive global issue affecting vulnerable individuals, violating their fundamental human rights. Investigations reveal that a significant number of HT cases are associated with online advertisements (ads), particularly in escort markets. Consequently, identifying and connecting HT vendors has become increasingly challenging for Law Enforcement Agencies (LEAs). To address this issue, we introduce IDTraffickers, an extensive dataset consisting of 87,595 text ads and 5,244 vendor labels to enable the verification and identification of potential HT vendors on online escort markets. To establish a benchmark for authorship identification, we train a DeCLUTR-small model, achieving a macro-F1 score of 0.8656 in a closed-set classification environment. Next, we leverage the style representations extracted from the trained classifier to conduct authorship verification, resulting in a mean r-precision score of 0.8852 in an open-set ranking environment. Finally, to encourage further research and ensure responsible data sharing, we plan to release IDTraffickers for the authorship attribution task to researchers under specific conditions, considering the sensitive nature of the data. We believe that the availability of our dataset and benchmarks will empower future researchers to utilize our findings, thereby facilitating the effective linkage of escort ads and the development of more robust approaches for identifying HT indicators.

![(i) Closed-Set Vendor Verification Task: A supervised-pretraining task that performs classification in a closed-set environment setting to verify unique vendor migrants across known markets (ii) Open-set Vendor Identification Task: A text-similarity task in open-set environment setting that utilizes embeddings from the pre-trained classifier to verify known vendors and identify potential-aliases (iii) Low-resource domain adaptation task: A knowledge-transfer task to adapt new domain knowledge and verify migrants in a closed-set environment setting across low-resource emerging markets.](Images/Screenshot from 2023-10-12 22-18-06.png)
