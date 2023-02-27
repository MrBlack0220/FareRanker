import copy
import os
import re
from typing import Union, Tuple

import numpy as np
from func_timeout import func_set_timeout

from data_preprocessors.language_processors import PythonProcessor
from data_preprocessors.transformations.transformation_base import TransformationBase

import io
processor_function = {
    "python": [PythonProcessor.condition_bugs]
}


class ConditionBugs(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(ConditionBugs, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "python": PythonProcessor.get_tokens,
        }
        self.final_processor = processor_map[self.language]
        
    @func_set_timeout(5)
    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_code, success = function(code, self)
        return modified_code,success

    


if __name__ == '__main__':
    import random
    python_code ="""
from sys import stdin, stdout
n,m,o = map(int,stdin.readline().split())
n= n+m
l=[]
for i in range(n):
	a=int(stdin.readline())
	if a in l and a not in l[:-1]:
		l.append(a)
	else:
		l.append(a)
	

l.sort()
stdout.write(str(len(l)) + '\n')
stdout.write(''.join(str(i) for i in l))"""
    input_map = {
        "python": ("python", python_code)
    }
    parser_path ="/home2/MyEXp/ranker_debug/parser/languages.so"
    for lang in ["python"]: 
        lang, code = input_map[lang]
        operandswap = ConditionBugs(
            parser_path, lang
        )
        print(lang)
        print(code)
        code, meta = operandswap.transform_code(code)
        print("============================================================================")
        print(code)
        print(meta)
        print("=" * 150)
        

