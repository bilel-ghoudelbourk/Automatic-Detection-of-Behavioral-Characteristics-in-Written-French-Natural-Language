
# Automatic Detection of Behavioral Characteristics in Written Natural Language

![Python Version](https://img.shields.io/badge/Python-3.11.2-blue)


Author: Bilel GHOUDELBOURK

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Context and Theoretical Framework](#context-and-theoretical-framework)
4. [Proposed Approach](#proposed-approach)
5. [Implementation using NER Method](#implementation-using-ner-method)
6. [Conclusion](#conclusion)
7. [References](#references)

## Problem Statement

HR departments often rely on cover letters to evaluate candidates. The objective is to utilize automatic text analysis and AI advancements to extract useful details from these letters, streamlining the selection process and ensuring accuracy. The dual goals are to find the best job fit and assess compatibility with the company's culture.

## Project Overview

In collaboration with Skilit company, this project aims to develop an automatic behavioral analysis tool. The project involves creating a prototype solution using pretrained language models from the Hugging Face platform.

## Context and Theoretical Framework

### Automatic Text Analysis

This involves the understanding, manipulation, and study of discourse. Coupled with AI technologies, it enables applications ranging from spell checking to complete human speech imitation.

### Behavioral Psychology

This branch of psychology focuses on studying human behaviors, helping to understand how individuals act and react in various situations. It's used to develop text analysis systems to extract personality and behavior information from writings.

### Fusion of Disciplines

NLP models and psychological models are used together to analyze texts and identify psychological traits using techniques like word frequency analysis, semantic analysis, and neural networks.

## Proposed Approach

### Key Terms

- **Fine-tuning**: Adapting a pretrained model to a specific task using additional annotated data.
- **Hugging Face Transformers**: An open-source library for implementing state-of-the-art NLP models.
- **CamemBERT**: A French version of BERT for improving NLP applications in French.
- **NER (Named Entity Recognition)**: Identifying and classifying named entities in text documents.

### Dataset

The dataset used is not publicly available and is provided by Skilit company.

### Data Augmentation

Controlled deformation of existing data to create similar yet distinct data examples, enhancing the training set size without additional data collection.

Method explanation: We take annotated words, calculate the word embeddings for each word, compute the centroids for each axis, and then determine the closest centroid for each word to evaluate whether it is close to its true centroid or another centroid. We then calculate the error.

After analyzing the error, we found that using this method for data augmentation does not yield satisfactory results.

### Fine-tuning NER Approach

Utilizing pretrained language models like CamemBERT, fine-tuned for the NER task to classify texts into specific behavioral axes.

## Implementation using NER Method

### Data Preprocessing

Splitting data into training, validation, and test sets. Proper tokenization and annotation alignment are crucial for effective fine-tuning.

### Fine-tuning

Training the CamemBERT model using the Trainer API from the Hugging Face library, adjusting hyperparameters for optimal performance.

### Testing

Evaluating the model on a test set to assess its performance using confusion matrices and classification reports.

## Conclusion

The project successfully applied advanced NLP techniques to detect behavioral traits in written text, demonstrating significant potential for real-world applications in HR. The collaboration with Skilit provided valuable insights into the practical implementation of AI technologies.

## References

1. Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É. V., ... & Sagot, B. (2020). CamemBERT: a Tasty French Language Model. *arXiv preprint arXiv:1911.03894*. Available at [https://arxiv.org/abs/1911.03894](https://arxiv.org/abs/1911.03894).
2. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *arXiv preprint arXiv:1603.01360*. Available at [https://arxiv.org/abs/1603.01360](https://arxiv.org/abs/1603.01360).
3. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Available at [https://www.aclweb.org/anthology/2020.emnlp-demos.6/](https://www.aclweb.org/anthology/2020.emnlp-demos.6/).
4. [Skilit Company](https://www.skilit.io/qui-sommes-nous)

