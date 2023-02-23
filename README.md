# NLPResearchScaffolding
Scaffold for NLP researcher to quickly set up the codebase

Author: Chenghao Yang (chenghao at uchicago dot edu)

## Usage
First install anaconda corresonpoding to your system architecture.  

Then just run the `run.sh` shell scripts and you will get the environment setup.

## Current Environment Package
1. Basic NLP toolkits (NLTK & Spacy), with their package data downloaded. 
1. Basic ML and Scientific Computing Toolkits (scipy, scikit-learn)
1. Pytorch (may add TensorFlow later if there is enough requests and good PRs :-) )
1. Huggingface Suites (Transformers, Datasets, Evaluate, Accelerate)
1. Large Language Models stuffs (Protobuf, SentencePiece)
1. Logging Library (loguru) and Vim Setup from [Ultimate Vimrc](https://github.com/amix/vimrc). 

## TO-DOs
1. Tries to provide some interface to easilly select which toolkit we want to install.
2. Tries to suppress all user checking ([y/n]) and make it fully automatic. 
