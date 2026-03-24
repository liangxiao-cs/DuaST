# DuaST: An Integrated Deep Learning Framework for Spatial Transcriptomics with Cross-Branch Interaction

![](https://github.com/liangxiao-cs/DuaST/blob/main/Fig1.png)

## Overview
The overall framework of DuaST. A. Dual-branch framework. The Spatial-Aware Branch captures neighborhood dependencies through graph-based encoding, and the Non-Spatial Branch extracts topology-agnostic features, providing two complementary views of the data that are subsequently coordinated through contrastive, adversarial, and attention-based interactions. While spatial relevance scoring operates by reconstructing gene expression with a learnable gene-wise weight matrix, leveraging contrastive and adversarial learning yet without involving the attention mechanism. B. Downstream tasks. DuaST’s tasks in spatial domain identification, multi-omics integration, and SVGs detection.

## Example
To run the project, **simply execute HBC.py directly**.
All code and functions are integrated into this single file for user convenience.
Dataset: Please go to https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0 to download.

Note:
1. Please replace all **file paths / data addresses** in the code with your own local paths before running.
2. The R package mclust is available andR_HOME is correctly configured if needed.


## Contact
You can contact us via liangxiao1@hnu.edu.cn.  

## Citation
If you find this repository useful, please cite our paper.
