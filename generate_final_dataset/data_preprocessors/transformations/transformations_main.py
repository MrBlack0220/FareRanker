import numpy as np
from typing import Dict, Callable
import copy
from data_preprocessors.transformations import (
     DeadCodeInserter, ForWhileTransformer,OperandSwap,PriorityTrans,VarRenamerBase
)
from data_preprocessors.transformations import (
    BooleanBugs,ConditionBugs,FunctionargsBugs,
InputBugs,IndexBugs,OperatorBugs,PriorityBugs,RangeBugs,VarmodifyBugs
)
import random
import func_timeout
class SemanticPreservingTransformation:
    def __init__(
            self,
            parser_path: str,
            language: str,
            transform_functions: Dict[Callable, int] = None,
    ):
        self.language = language
        self.transform_functions = transform_functions
        self.transformations = []  
        for t in self.transform_functions:   
            for _ in range(self.transform_functions[t]):   
                self.transformations.append(t(parser_path=parser_path, language=language))
                
                
    def new_lines_general(self,code, ratio=0.5):
        """ Adding new lines after every lines
        """
        try:
            partial_code = code
            num_lines = 0
            added_lines = 0
            new_partial_code = "\n"
            for ch in partial_code:
                new_partial_code += ch
                if ch == "\n": 
                    num_lines += 1
                    if random.random() < ratio:
                        new_partial_code += "\n"
                        added_lines += 1
            new_partial_code = new_partial_code+ '\n'
            return new_partial_code
        except:
            return code
    
    def delete_lines_or_tokens(self,code):
        try:
            code_lines = code.split("\n")
            code_lines_with_content =[i for i in code_lines if i.replace(" ", "").replace("\t", "").replace("\n", "") != ""]
            if len(code_lines) > 1:
                del_line = random.choice(code_lines_with_content)
                code_lines.remove(del_line)
                return "\n".join(code_lines)
            else:
                code_list = code.split()
                num = len(code_list)
                delidx = random.randint(0,num-1)
                code_list = code_list[:delidx]+code_list[delidx+1:]
                return " ".join(code_list)
        except:
            return code        
    
    
    def transform_code(
            self,
            code: str,
            mode,
            positive_max_num,
            negative_max_num
    ):  
        if mode == "positives":
            positives = []
            transformation_name = []
            for i in range(positive_max_num):
                transformed_code= None
                indices = list(range(len(self.transformations)))
                np.random.shuffle(indices)  
                success = False
                while not success and len(indices) > 0:
                    si = np.random.choice(indices) 
                    indices.remove(si)
                    t = self.transformations[si]  
                    try:
                        transformed_code, success = t.transform_code(code)  
                    except func_timeout.exceptions.FunctionTimedOut:
                        print("timeout!")
                        transformed_code = code
                        success = False
                    except:
                        transformed_code = code
                        success = False
                    if success:
                        transformation_name.append(type(t).__name__)
                        positives.append(transformed_code)
                        
            if len(positives) < (positive_max_num):
                for num in range(positive_max_num-len(positives)):
                    positives.append(self.new_lines_general(code))
                    transformation_name.append("AddNewLines")
            
            assert len(transformation_name) == positive_max_num   
            assert len(positives) == positive_max_num       
            return positives,transformation_name,True    
            
            
        if mode == "negatives":
            negatives = []
            transformation_name = []
            for i in range(negative_max_num):
                transformed_code= None
                indices = list(range(len(self.transformations)))
                np.random.shuffle(indices)  
                success = False
                while not success and len(indices) > 0:
                    si = np.random.choice(indices)  
                    indices.remove(si)
                    t = self.transformations[si]  
                    try:
                        transformed_code, success = t.transform_code(code)   
                    except:
                        transformed_code = code
                        success = False
                    if success:
                        transformation_name.append(type(t).__name__)
                        negatives.append(transformed_code)
            if len(negatives) < (negative_max_num):
                for num in range(negative_max_num-len(negatives)):
                    negatives.append(self.delete_lines_or_tokens(code))
                    transformation_name.append("DelRanLineOrToken")
                    
            assert len(negatives) == negative_max_num
            assert len(transformation_name) == negative_max_num
            return negatives,transformation_name,True
    
    