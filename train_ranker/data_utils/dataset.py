from torch.utils.data import Dataset, DataLoader
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
import pickle as pkl
import io
from typing import Optional
from dataclasses import dataclass, field
import sys
from transformers import (HfArgumentParser,set_seed)

class ReRankingDataset_eval(Dataset):
    def __init__(self, file,args = None):
        self.file = file
        self.test_data = pkl.load(open(file,'rb'))
        self.args = args
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        sample = self.test_data[idx]
        return sample


class ReRankingDataset_train(Dataset):
    def __init__(self, dir ,tokenizer,args = None):

        self.dir = dir 
        self.args = args
    def read_data(self,epoch):
        if epoch%5 == 0: 
            num = 5
        else:
            num = epoch % 5  
        file = os.path.join(self.dir,str(num)+"_epoch.pkl")
        f = open(file,'rb')
        self.raw_data = pkl.load(f)
        f.close()
        
 
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample_ = self.raw_data[idx]  
        sample = {}
        sample['step1_data'] = sample_['step1_data'][:self.args.step1_max_num]
        sample['step2_data'] = sample_['step2_data'][:self.args.step2_max_num]
        return sample



   
    
    
    
