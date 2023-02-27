from dataclasses import dataclass
import torch
import numpy as np
from collections import Counter
from dataclasses import dataclass
import random
class compute_ranked_pass_k:
    def __call__(self, all_ids,all_preds,all_results):
        eval_preds = []    
        for i in range(0,len(all_ids),100):
            assert len(set(all_ids[i:i+100])) == 1
            result = []
            logit = all_preds[i:i+100]
            label = all_results[i:i+100]
            ordered_list = np.argsort(logit) 
            for j in ordered_list:
                result.append(label[j])
            eval_preds.append(result)
        eval_preds = np.array(eval_preds)
        best_1 = eval_preds[:,-1]
        sum_1 = np.sum(best_1, axis=0)
        ranked_pass_1 = sum_1/eval_preds.shape[0]
        best_2 = eval_preds[:,-2:]
        sum_2 = eval_preds.shape[0]-np.sum(np.where(np.sum(best_2, axis=1),0,1))  
        ranked_pass_2 = sum_2/eval_preds.shape[0]
        best_5 = eval_preds[:,-5:]
        sum_5 = eval_preds.shape[0]-np.sum(np.where(np.sum(best_5, axis=1),0,1))
        ranked_pass_5 = sum_5/eval_preds.shape[0]
        eval_result = {}
        eval_result["ranked_pass@1"] = round(ranked_pass_1,6)
        eval_result["ranked_pass@2"] = round(ranked_pass_2,6)
        eval_result["ranked_pass@5"] = round(ranked_pass_5,6)
        return eval_result


