# DuaST: An Integrated Deep Learning Framework for Spatial Transcriptomics with Cross-Branch Interaction

![](https://github.com/liangxiao-cs/DuaST/blob/main/Fig1.png)

## Overview
The overall framework of DuaST. A. Dual-branch framework. The Spatial-Aware Branch captures neighborhood dependencies through graph-based encoding, and the Non-Spatial Branch extracts topology-agnostic features, providing two complementary views of the data that are subsequently coordinated through contrastive, adversarial, and attention-based interactions. While spatial relevance scoring operates by reconstructing gene expression with a learnable gene-wise weight matrix, leveraging contrastive and adversarial learning yet without involving the attention mechanism. B. Downstream tasks. DuaST’s tasks in spatial domain identification, multi-omics integration, and SVGs detection.

Please also make sure that:
R is installed
the R package mclust is available
R_HOME is correctly configured if needed

### Current Status
The paper has been **accepted** by [***Briefings in Bioinformatics***].
Starting from **23/03/2026**, we will release the full codebase within 10 days, including:

1. Release the complete, runnable codebase
2. Add detailed and comprehensive code comments
3. Standardize all implementations for better readability and reproducibility
4. Provide full documentation 

### Contact
You can contact us via liangxiao1@hnu.edu.cn  
Welcome to communicate and discuss related research!
