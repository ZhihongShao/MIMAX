## Introduction

* This directory includes code for experiments on WikiSQL (using the hard train set specifically).

## Requirements

`pip install -r requirements.txt`

## Quick Start

* Preprocessed version of WikiSQL as well as our model trained on the hard train set can be downloaded [here](https://cloud.tsinghua.edu.cn/f/13982c09e1084685a3a8/?dl=1).
* Run the script `train.sh` for training and `infer.sh` for inference. The only argument needed by the script is the name of the training algorithm, which should be chosen from `[mml, hard-em, hard-em-thres, mimax]` where `mimax` denotes our training method.