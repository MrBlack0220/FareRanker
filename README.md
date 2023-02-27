# SemRanker
This is an implementation of the model described in: Semantics-Aware Rankers for Code Generation through Contrastive Learning

In this paper, we propose SemRanker, a novel contrastive learning based Semantic-aware Ranker for code generation. We make three improvements through contrastive learning to understand the code semantics. 

First, we introduce two methods, Code Generation Models’ Predictions and Error Injction Heuristics, to automatically generate high-quality hard negative samples.

Then, we iteratively train SemRanker with two contrastive objects, a discrimination object and an alignment object, injecting the code’s semantic knowledge into the model. 

Finally, in inference, we arrange the ranking scores among the origin code candidate and its semantic-preserving transformations to get the final score, which is more robust and accurate.

## Requirements
* python3
* tree-sitter
* torch
* transformers==4.8.1

## Dataset
You should first download the [APPS](https://github.com/hendrycks/apps) dataset.

## Obtain the Code Generation Model
We use the [checkpoint](https://console.cloud.google.com/storage/browser/sfr-coderl-research/codet5_finetuned_codeRL) for CodeRL model. And for GPT-Neo-125M, please first finetune the model for 2 epoch by running:
```
cd finetune
bash train.sh
```
## Sample Code Candidates
You can sample code candidates for APPS training set and test set by running (please change the arguments for dataset path bu yourself):
```
cd ../sample
bash generate_coderl_apps.sh
```
## Run the Unit Tests

To execute the unit tests and obtain test outcomes, we adapt the official implementation of the [APPS benchmark](https://github.com/hendrycks/apps). You can run the following commands by configuring the parameters as you need:
```
cd ../run_test
bash test_one_solution.sh
```

## Obtain the Final Datasets
Then, you can build the positive and negative samples and get the final training and test datasets by runing:
```
cd ../data_preprocess
bash data_preprocess.sh
cd ../generate_final_dataset
python dataset.py
```
## Train SemRanker
Finally, you can train Semranker by running:
```
cd ../train_ranker
bash run_train.sh
```
