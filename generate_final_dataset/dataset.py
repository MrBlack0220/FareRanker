import os
import json
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
import time
from transformers import RobertaTokenizer
import random
import pickle
import copy
import logging
import math
import copy
import argparse
import json
import multiprocessing
import os
import pickle
import random
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
from tqdm import tqdm
from data_preprocessors.transformations import (
    SemanticPreservingTransformation, DeadCodeInserter,
    ForWhileTransformer, OperandSwap, VarRenamerBase,PriorityTrans
)
from data_preprocessors.transformations import (
    BooleanBugs,ConditionBugs,FunctionargsBugs,
InputBugs,IndexBugs,OperatorBugs,PriorityBugs,RangeBugs,VarmodifyBugs
)
import io
from typing import Optional
from dataclasses import dataclass, field
import sys
from transformers import (HfArgumentParser,set_seed)

def encode_seq(tokenizer, seq):
    """
    encode the tokenized or untokenized sequence to token ids
    """
    if seq == [] or seq == "":
        return []
    else:
        return tokenizer.encode(seq, add_special_tokens=False)
    
def create_transformers_from_conf_file(processing_conf,mode):
    if mode == "positives":
        classes = [DeadCodeInserter,ForWhileTransformer,OperandSwap,VarRenamerBase,PriorityTrans]
        transformers = {
            c: processing_conf[c.__name__] for c in classes
        }
    else:
        classes = [BooleanBugs,ConditionBugs,FunctionargsBugs,InputBugs,IndexBugs,OperatorBugs,PriorityBugs,RangeBugs,VarmodifyBugs]
        transformers = {
            c: processing_conf[c.__name__] for c in classes
        }
    return transformers



class ExampleProcessor:
    def __init__(
            self,
            language,
            parser_path,
            transformation_config,
            mode,
            positive_max_num,
            negative_max_num
    ):
        self.language = language
        self.parser_path = parser_path
        self.transformation_config = transformation_config
        self.mode = mode
        self.positive_max_num = positive_max_num
        self.negative_max_num = negative_max_num
        transformers = create_transformers_from_conf_file(self.transformation_config,self.mode)
        self.example_transformer = SemanticPreservingTransformation(   
            parser_path=self.parser_path, language=self.language, transform_functions=transformers
        )

    def process_positives_example(self, data):
        success = 0
        try:    
            transformed_code, used_transformer,success = self.example_transformer.transform_code(data["positive"],self.mode,self.positive_max_num,self.negative_max_num)
            if success:
                return {"id":data["id"],
                        "prompt":data["prompt"],
                        "positive":data["positive"],
                        "tf_positive":transformed_code,
                        "transformer":used_transformer}
            else:
                print("Not success.try Stopping parsing for: \n", data["positive"])
                print("problem:id:",str(data["id"]))
                return -1
        except Exception as e:
            print("Expection occur.Stopping parsing for \n", data["positive"])
            print("problem id:",str(data["id"]))
            print("Exception:",e)
            return -1

    def process_negatives_example(self, data):
        success = 0
        try:    
            transformed_code,transformed_type,success = self.example_transformer.transform_code(data["positive"],self.mode,self.positive_max_num,self.negative_max_num)
            if success:
                return {"id":data["id"],
                        "prompt":data["prompt"],
                        "positive":data["positive"],
                        "tf_positive":data["tf_positive"],
                        "tf_negatives":transformed_code,
                        "transformer":transformed_type}
            else:
                print("Not success.try Stopping parsing for: \n", data["positive"])
                print("problem:id:",str(data["id"]))
                return -1
        except Exception as e:
            print("Expection occur.Stopping parsing for \n", data["positive"])
            print("problem id:",str(data["id"]))
            print("Exception:",e)
            return -1
    
    def process_test_example(self, data):
        success = 0
        try:   
            transformed_code, used_transformer,success = self.example_transformer.transform_code(data["candidate"],self.mode,self.positive_max_num,self.negative_max_num)
            if success:
                return {
                        "id":data["task_id"],
                        "prompt":data["prompt"],
                        "candidate":data["candidate"],
                        "tf_candidate":transformed_code,
                        "result":data["result"],
                        "transformer":used_transformer
                        }
            else:
                print("Not success.try Stopping parsing for: \n", data["positive"])
                print("problem:id:",str(data["id"]))
                return -1
        except Exception as e:
            print("Expection occur.Stopping parsing for \n", data["positive"])
            print("problem id:",str(data["id"]))
            print("Exception:",e)
            return -1


class CreatTestData:
    def __init__(self,file,tokenizer, args = None):
        self.test_data = json.load(open(file,'r'))
        print(len(self.test_data))
        self.configuration = json.load(open(args.test_processing_config_file))
        self.id_list = list(self.test_data.keys())  # 每个问题的序号
        self.args = args
        self.tok = tokenizer
        self.pad_token_id = self.tok.pad_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        
  
    def read_data(self):
        '''
            cache the data in memory
        '''
        data = []
        for i in tqdm(self.id_list): 
            problem = self.test_data[i]
            prompt = encode_seq(self.tok,problem['prompt'])[:self.args.max_source_length] 
            candidates = problem['code']
            results = problem['result']
            for index in range(len(candidates)):
                sample = {
                    "task_id":i,
                    "prompt":prompt,
                    "candidate":candidates[index].replace(";\n", "\n").replace(";", "\n"),
                    "result":results[index]
                }
                data.append(sample)
        
        example_processor_p = ExampleProcessor(
            language="python",
            parser_path=self.args.parser_path,
            transformation_config=self.configuration["transformers_positive"],
            mode = "positives",
            positive_max_num = self.args.test_positive_max_num,
            negative_max_num = self.args.test_negative_max_num
        )
    
        print(40*"="+"Start to generate positive samples"+40*"=")
        
        
        
        data_ids = []
        transformers = []
        for idx,sample in enumerate(tqdm(data)):
            sample = example_processor_p.process_test_example(sample)
            if sample==-1:
                continue
            problem_id = sample['id']
            prompt_id =  sample['prompt']
            transformers.append(sample["transformer"]) 
            transformation_cands = [sample['candidate']]+sample['tf_candidate']
            transformation_cands_ids = []
            for t in transformation_cands:
                mode_id = 6   # <encoder>
                input_id = [self.tok.cls_token_id,mode_id,self.tok.sep_token_id]
                input_id += prompt_id
                input_id += [self.tok.sep_token_id]
                input_id += encode_seq(self.tok,t)[:self.args.max_candidate_length]
                input_id += [self.tok.sep_token_id]
                transformation_cands_ids.append(input_id)
            
            data_ids.append({
                    "ids": problem_id,
                    "input_ids":transformation_cands_ids,
                    "result":sample["result"],
                    }) 
            
        used_transformers = {}
        success = 0
        for r in transformers:
            for s in r:
                if s not in used_transformers.keys():
                    used_transformers[s] = 0
                used_transformers[s] += 1
            success += 1
            
        print(40*"="+"Positive samples generation finish!"+40*"=")
        print(
            f"""
                Total   : {len(data)}, 
                Success : {success},
                Failure : {len(data) - success}
                Stats   : {json.dumps(used_transformers, indent=4)}
                """
        )
        del example_processor_p
        
  
        return data_ids





class CreatEpochTrainData:
    def __init__(self,file,tokenizer, args = None):
        self.raw_data = json.load(open(file,'r'))
        self.configuration = json.load(open(args.train_processing_config_file))
        self.args = args
        self.tok = tokenizer
        self.pad_token_id = self.tok.pad_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        self.train_data = self.read_data()  
        
  
    def read_data(self):
        self.id_list = list(self.raw_data.keys()) 
        data = []
        for i in tqdm(self.id_list):  
            problem = self.raw_data[i]
            prompt = problem['prompt']
            if len(problem['origin_p']+problem['generate_p']) == 0:
                continue 
            if len(problem['generate_n']) < self.args.max_num-1:
                continue
            for positive in (problem['origin_p']+problem['generate_p']):
                sample = {
                    "id":i,
                    "prompt":prompt,
                    "positive":positive.replace(";\n", "\n").replace(";", "\n")
                }
                data.append(sample)
        return data
    
    
    def tf_positives(self,data,example_processor):
        this_epoch_data = []
        for sample in tqdm(data):
            sample = example_processor.process_positives_example(sample)
            if sample == -1:
                continue
            this_epoch_data.append(sample)
        success =  0   
        used_transformers = {}
        for sample in this_epoch_data:
            if isinstance(sample, int) and sample == -1:
                continue
            else:
                for s in sample["transformer"]:
                    if s not in used_transformers.keys():
                        used_transformers[s] = 0
                    used_transformers[s] += 1
                success += 1
        print(40*"="+"Positive samples generation finishes!"+40*"=")
        print(
            f"""
                Total   : {len(data)}, 
                Success : {success},
                Failure : {len(data) - success}
                Stats   : {json.dumps(used_transformers, indent=4)}
                """
        )
        return this_epoch_data
      
              
    def tf_negatives(self,data,example_processor):

        this_epoch_data = []
        for idx,sample in enumerate(tqdm(data)):
            sample = example_processor.process_negatives_example(sample)
            if sample ==-1:
                continue
            this_epoch_data.append(sample)
            
        used_transformers = {}
        success = 0
        for sample in this_epoch_data:
            if isinstance(sample, int) and sample == -1:
                continue
            else:
                for s in sample["transformer"]:
                    if s not in used_transformers.keys():
                        used_transformers[s] = 0
                    used_transformers[s] += 1
                success += 1

        print("="*100)
        print("Negative sample genaration finishes!")
        print(
            f"""
                Total   : {len(data)}, 
                Success : {success},
                Failure : {len(data) - success}
                Stats   : {json.dumps(used_transformers, indent=4)}
                """
        )
        return this_epoch_data
    
    
    def reset_epoch(self,epoch_num):
        print(40*"="+"epoch{},reset dataset".format(epoch_num)+40*"=")
        print(40*"="+"start to generate positive samples"+40*"=")
        example_processor_p = ExampleProcessor(
            language="python",
            parser_path=self.args.parser_path,
            transformation_config=self.configuration["transformers_positive"],
            mode = "positives",
            positive_max_num = self.args.train_positive_max_num,
            negative_max_num = self.args.train_negative_max_num
        )
        
        this_epoch_data = self.tf_positives(data= self.train_data,example_processor=example_processor_p)  
        del example_processor_p
        
        print(40*"="+"start to generate negative samples"+40*"=")

        example_processor_n = ExampleProcessor(
            language="python",
            parser_path=self.args.parser_path,
            transformation_config=self.configuration["transformers_negative"],
            mode = "negatives",
            positive_max_num = self.args.train_positive_max_num,
            negative_max_num = self.args.train_negative_max_num
        )
        this_epoch_data = self.tf_negatives(data= this_epoch_data,example_processor=example_processor_n)
        del example_processor_n
    
        
        print(40*"="+"start to tokenize data"+40*"=")
        epoch_data_ids = []
        for sample in tqdm(this_epoch_data):
            problem_id = sample['id']
            prompt_id = encode_seq(self.tok,sample['prompt'])[:self.args.max_source_length]
            all_negetives = self.raw_data[problem_id]['generate_n']
            
            generator_cands = [sample['positive']] + random.sample(all_negetives, self.args.max_num-1)   # 选出15个
            generator_cands_ids = []
            for g in generator_cands:
                mode_id = 6   # <encoder>
                input_id = [self.tok.cls_token_id,mode_id,self.tok.sep_token_id]
                input_id += prompt_id
                input_id += [self.tok.sep_token_id]
                input_id += encode_seq(self.tok,g)[:self.args.max_candidate_length]
                input_id += [self.tok.sep_token_id]
                generator_cands_ids.append(input_id)   
            transformation_cands = [sample['positive']]+[sample['tf_positive']]+sample['tf_negatives']
            transformation_cands_ids = []
            for t in transformation_cands:
                mode_id = 6   # <encoder>
                input_id = [self.tok.cls_token_id,mode_id,self.tok.sep_token_id]
                input_id += prompt_id
                input_id += [self.tok.sep_token_id]
                input_id += encode_seq(self.tok,t)[:self.args.max_candidate_length]
                input_id += [self.tok.sep_token_id]
                transformation_cands_ids.append(input_id) 
            epoch_data_ids.append({"step1_data": generator_cands_ids,"step2_data":transformation_cands_ids})
            
        print(50*"="+"This epoch data generation finishes!"+50*"=")
        
        return epoch_data_ids





@dataclass
class DataTrainingArguments:
    model_name: str = field(
        default="coderl",
        metadata={
            "help": "model name"
        },
    )
    epoch_num: int = field(
        default=5,
        metadata={
            "help": "epoch num"
        },
    )
    
    train_positive_max_num : Optional[int] = field(
        default=1,
        metadata={
            "help": "positive sample num for training"
        },
    )

    train_negative_max_num : Optional[int] = field(
        default=6,
        metadata={
            "help": "error injection negative sample num for training"
        },
    )

    test_positive_max_num : Optional[int] = field(
        default=2,
        metadata={
            "help": "augmentation num for test"
        },
    )
    
    test_negative_max_num : Optional[int] = field(
        default=0,
        metadata={
            "help": "not use"
        },
    )
    max_num : Optional[int] = field(
        default=8,
        metadata={
            "help": "disc loss sample bank length"
        },
    )

    max_source_length : Optional[int] = field(
        default=512,
        metadata={
            "help": "max source length"
        },
    )

    max_candidate_length : Optional[int] = field(
        default=600,
        metadata={
            "help": "max candidate length"
        },
    )

    train_data_path: Optional[str] = field(
        default="../data_preprocess/datasets/coderl/train.json"
    )

    dev_data_path: Optional[str] = field(
        default="../data_preprocess/datasets/coderl/eval.json"
    )

    test_data_path: Optional[str] = field(
        default="../data_preprocess/datasets/coderl/test.json"
    )
    parser_path : Optional[str] = field(
        default="parser",
        metadata={
            "help": "tree-sitter parser_path"
        },
    )
    
    train_processing_config_file : Optional[str] = field(
        default="data_config/train_data_processing_config.json"
    )
    
    test_processing_config_file : Optional[str] = field(
        default="data_config/test_data_processing_config.json"
    ) 
    
    
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()[0]
    tokenizer = RobertaTokenizer.from_pretrained(
            "microsoft/unixcoder-base"
        )
    
    tokenizer.add_tokens(["<mask0>"],special_tokens=True)
    
    train_dataset_process = CreatEpochTrainData(file = data_args.train_data_path,tokenizer=tokenizer,args = data_args)  
    for i in range(1,data_args.epoch_num):
        epoch_data_ids = train_dataset_process.reset_epoch(i)
        if not os.path.exists('datasets/%s/train/genaration_%s_translation_%s_tokenizedata'%(data_args.model_name,data_args.max_num,data_args.train_negative_max_num+2)):
            os.makedirs('datasets/%s/train/genaration_%s_translation_%s_tokenizedata'%(data_args.model_name,data_args.max_num,data_args.train_negative_max_num+2))
        with open('datasets/%s/train/genaration_%s_translation_%s_tokenizedata/%s_epoch.pkl'%(data_args.model_name,data_args.max_num,data_args.train_negative_max_num+2,i),"wb") as f:
            pickle.dump(epoch_data_ids, f)
    
    
    eval_data_process = CreatTestData(file = data_args.dev_data_path,tokenizer=tokenizer,args = data_args)
    data_ids = eval_data_process.read_data()
    if not os.path.exists('datasets/%s/eval'%(data_args.model_name)):
        os.makedirs('datasets/%s/eval'%(data_args.model_name))
    with open('datasets/%s/eval/translation_%s_tokenizedata.pkl'%(data_args.model_name,data_args.test_positive_max_num),"wb") as f:
        pickle.dump(data_ids, f)
    
    
    test_data_process = CreatTestData(file = data_args.test_data_path,tokenizer=tokenizer,args = data_args)
    data_ids = test_data_process.read_data()
    if not os.path.exists('datasets/%s/test'%(data_args.model_name)):
        os.makedirs('datasets/%s/test'%(data_args.model_name))
    with open('datasets/%s/test/translation_%s_tokenizedata.pkl'%(data_args.model_name,data_args.test_positive_max_num),"wb") as f:
        pickle.dump(data_ids, f)
        


