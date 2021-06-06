## Introduction

* This directory includes code for experiments on Quasar-T and WebQuestions

## Quick Start

* Preprocessed datasets and our models can be downloaded [here](https://drive.google.com/PLACEHOLDER).
* Run the script `preprocess.sh` for data preprocessing (no needed if you used our preprocessed data), `train.sh` for training, and `infer.sh` for inference.
  * The only argument needed by `preprocess.sh` is the name of the target dataset which should be chosen from `[quasart, webquestions]`.
  * Both `train.sh` and `infer.sh` require two arguments, i.e., the name of the target dataset chosen from `[quasart, webquestions]` and the name of the training algorithm chosen from `[mml, hard-em, hard-em-thres, mimax]`.