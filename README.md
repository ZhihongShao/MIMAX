# A Mutual Information Maximization Approach for Weakly Supervised Question Answering

## Introduction

Weakly supervised question answering usually has only the final answers as supervision signals while the correct solutions to derive the answers are not provided. This setting gives rise to the ***spurious solution problem***: there may exist many spurious solutions that coincidentally derive the correct answer, but training on such solutions can hurt model performance (e.g., producing wrong solutions or answers). For example, for discrete reasoning tasks as on DROP, there may exist many equations to derive a numeric answer, and typically only one of them is correct. Previous learning methods mostly filter out spurious solutions with heuristics or using model confidence, but do not explicitly exploit the semantic correlations between a question and its solution. 
In this paper, to alleviate the spurious solution problem, we propose to explicitly exploit such semantic correlations by maximizing the mutual information between question-answer pairs and predicted solutions. Extensive experiments on four question answering datasets show that our method significantly outperforms previous learning methods in terms of task performance and is more effective in training models to produce correct solutions.

```
@inproceedings{Shao2021MIMAX,
  author = {Zhihong Shao and Lifeng Shang and Qun Liu and Minlie Huang},
  title = {A Mutual Information Maximization Approach for Weakly Supervised Question Answering},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year = {2021}
}
```

## Quick Start

* This is an example repository for applying our training method to weakly supervised question answering. Details of where to download the data and our models as well as how to run our code can be found in each individual directory.

## Contact

* For any question, please contact [Zhihong Shao](szh19@mails.tsinghua.edu.cn) or post Github issue.