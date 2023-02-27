import json
import pickle as pkl
import os
from tqdm import tqdm
import random
import re
from io import StringIO
import tokenize
import glob
from dataset_lm.reindent import run as run_reindent
import time
import io

def DeleteComment(s): 
    s = re.sub(r'(#.*)', '', s)
    s= re.sub(r'(\'\'\')[\s\S]*?(\'\'\')', "", s, re.S)
    s= re.sub(r'(\"\"\")[\s\S]*?(\"\"\")', "", s, re.S)
    return s

def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )
    return ret.getvalue()




def process_raw_data(args):
    samples = {}  
    samples_1_10 = {}  
    samples_1_10_idx = [] 
    negetive_num = {}
    
    if os.path.isdir(args.train_unit_test_results_dir):
        files = sorted(os.listdir(args.train_unit_test_results_dir)) 
        num = len(files)       
        
    if os.path.isdir(args.apps_train_dir):
        apps_train_files = sorted(os.listdir(args.apps_train_dir))
    
    remove = 1
    # TODO
    for id in tqdm(range(num)):  
        generate_p = []
        generate_n = []
        origin_p = []
        
        
        if os.path.exists(os.path.join(args.apps_train_dir,apps_train_files[int(files[id][:-4])],"solutions.json")):  
            assert int(apps_train_files[int(files[id][:-4])]) == int(files[id][:-4])
            origin_sols_name = os.path.join(args.apps_train_dir,apps_train_files[int(files[id][:-4])],"solutions.json")  
            origin_p_ = json.load(open(origin_sols_name,'r')) 
            if len(origin_p_)>0:
                for op in origin_p_:
                    origin_p.append(reindent_code(DeleteComment(op)))   # we reindent code to have a same indent format in case that rankers learn from indent types (predictions always use \t but ground truth always us '    ')
        
        generate_sols_name = os.path.join(args.train_generate_sols_dir,files[id][:-4]+".json")  
        generate_sols_data = json.load(open(generate_sols_name,'r'))
        prompt = generate_sols_data[str(files[id][:-4])]['prompt'] 
        candidates = generate_sols_data[str(files[id][:-4])]['code']  
        
        test_result_name = files[id]  
        test_result_data = pkl.load(open(os.path.join(args.train_unit_test_results_dir,test_result_name),'rb'))  
        candidates_results = test_result_data[int(files[id][:-4])]["results"] 
        
        for can_dix , candidate in enumerate(candidates): 
            if -1 in candidates_results[can_dix] or -2 in candidates_results[can_dix] or False in candidates_results[can_dix] or len(candidates_results[can_dix])==0:
                generate_n.append(candidate)
            else:
                generate_p.append(candidate)
        assert  len(generate_n) + len(generate_p) == 100
        
        if  90 <= len(generate_n) < 100:
            samples_1_10_idx.append(int(files[id][:-4]))
            
        # delete the same code
        positives_uncopy = origin_p + ["SEPSEPSEP"] + generate_p
        positives = sorted(list(set(positives_uncopy)), key=positives_uncopy.index)
        origin_p = positives[:positives.index("SEPSEPSEP")]
        generate_p = positives[positives.index("SEPSEPSEP")+1:]
        negetive_num[int(files[id][:-4])]=len(generate_n)    
        generate_n = list(set(generate_n))
        
        sample = {
            'prompt': prompt,
            'origin_p':origin_p,
            'generate_p':generate_p,
            'generate_n':generate_n,
        }
        
        samples[int(files[id][:-4])] = sample
    
    return samples, samples_1_10_idx, negetive_num    


    
    
def creat_train_and_eval_dataset(all_samples,samples_1_10_idx,negetive_num,args):
    all_key  =  list(all_samples.keys())
    eval_dataset = {}
    
    if args.model_name == "coderl":
        # TODO：debug
        index = random.sample(samples_1_10_idx, 500)  
        eval_problem_list = {}
        eval_problem_list["coderl"] = index
        if not os.path.exists('%s'%(args.save_dataset_dir)):
            os.makedirs('%s'%(args.save_dataset_dir))
        with open('%s/eval_problem_list.json'%(args.save_dataset_dir), 'w', encoding='utf-8') as f:
            json.dump(eval_problem_list, f)
    else:
        index = json.load(open('%s/eval_problem_list.json'%(args.save_dataset_dir),'r'))["coderl"]  
              
    for j in index:
        eval_data = json.load(open(os.path.join(args.train_generate_sols_dir,str(j)+".json"),'r'))
        eval_data = eval_data[str(j)]
        result_data = pkl.load(open(os.path.join(args.train_unit_test_results_dir,str(j)+".pkl"),'rb'))
        candidate_results = result_data[j]["results"]  
        fina_result = []
        for candidate_result in candidate_results:
            if -1 in candidate_result or -2 in candidate_result or False in candidate_result or len(candidate_result)==0:
                fina_result.append(False)
            else:
                fina_result.append(True)
        eval_data['result'] = fina_result
        eval_dataset[j] = eval_data  
        del all_samples[j]  
        del negetive_num[j]
    assert len(list(eval_dataset.keys()))+len(list(all_samples.keys())) == len(all_key)
    
    if not os.path.exists('%s/%s'%(args.save_dataset_dir,args.model_name)):
        os.makedirs('%s/%s'%(args.save_dataset_dir,args.model_name))
    with open('%s/%s/eval.json'%(args.save_dataset_dir,args.model_name), 'w', encoding='utf-8') as f:
        json.dump(eval_dataset, f)

    with open('%s/%s/train.json'%(args.save_dataset_dir,args.model_name), 'w', encoding='utf-8') as f:
        json.dump(all_samples, f)
        


def creat_test_dataset(args):
    test_dataset = {}
    count=0
    problems = sorted(glob.glob(args.apps_test_dir + '/*')) 
    for problem_idx, problem in enumerate(tqdm(problems)): 
        problem_id = int(problem.split('/')[-1])
        test_data = json.load(open(os.path.join(args.test_generate_sols_dir,str(problem_id)+".json"),'r'))
        test_data = test_data[str(problem_id)]
        result_data = pkl.load(open(os.path.join(args.test_unit_test_results_dir,str(problem_id)+".pkl"),'rb'))
        candidate_results = result_data[problem_id]["results"]  
        fina_result = []
        for candidate_result in candidate_results:
            if -1 in candidate_result or -2 in candidate_result or False in candidate_result or len(candidate_result)==0:
                fina_result.append(False)
            else:
                fina_result.append(True)
        test_data['result'] = fina_result
        if 1<=sum(fina_result):
            count+=1
            test_dataset[problem_id] = test_data  
    if not os.path.exists('%s/%s'%(args.save_dataset_dir,args.model_name)):
        os.makedirs('%s/%s'%(args.save_dataset_dir,args.model_name))
    with open('%s/%s/test_part.json'%(args.save_dataset_dir,args.model_name), 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f)




def main(args):
    print(args)
    samples, samples_1_10_idx, negetive_num = process_raw_data(args)
    creat_train_and_eval_dataset(samples,samples_1_10_idx,negetive_num,args)
    creat_test_dataset(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="dataset")
    parser.add_argument("--model_name",default="coderl",type=str, help="Generator model name"),
    parser.add_argument("--save_dataset_dir",default="datasets",type=str, help="Path to store splited train/test/eval datasets"),
    parser.add_argument("--train_unit_test_results_dir", default="../run_test/outputs/coderl-train", type=str, help="Path to train dataset unit test result.")
    parser.add_argument("--test_unit_test_results_dir", default="../run_test/outputs/coderl-test", type=str, help="Path to test dataset unit test result.")
    parser.add_argument("--apps_train_dir", default="root/APPS/train",type=str, help="Path to raw APPS train dataset")
    parser.add_argument("--apps_test_dir", default="root/APPS/APPS/test",type=str, help="Path to raw APPS test dataset")
    parser.add_argument("--train_generate_sols_dir", default="../sample/outputs/coderl-train",type=str, help='Path to apps train dataset generated programs')
    parser.add_argument("--test_generate_sols_dir", default="../sample/outputs/coderl-test",type=str, help='Path to apps test dataset generated programs')
    parser.add_argument("-d", "--debug", default =False, help='test in debugging mode with printout messages')
    args = parser.parse_args()
    main(args)
    
    