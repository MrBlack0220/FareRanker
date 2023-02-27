import copy
import os
import re
from typing import Union, Tuple
from func_timeout import func_set_timeout
import numpy as np
import time
import random
from data_preprocessors.language_processors import (
    PythonProcessor,
)
from data_preprocessors.transformations.transformation_base import TransformationBase

python_code ="for _ in range(int(input())):\n\tn,p = map(int,input().split())\n\tl=list(map(int,input().split()))\n\ta=c=0\n\tfor i in l:\n\t if i >= int(p/2):\n\t\ta +=1 \n\t elif i <= int(p/10):\n\t\tc+= 1\n\tprint('yes' if(a==1 and c==2) else 'no') \n"
def beautify_python_code(tokens):
    """ A customized beautify function for python.
    NatGEN transformation will return a list of perturbed tokens, 
    we need to concatenate them together into a complete code.
    This is done before black normalization.
    """
    indent_count = 0
    code = ""
    i = 0
    new_tokens = []
    cat = False # if the next token should be concatenated to the previous one
    for ti, token in enumerate(tokens):
        if cat:
            cat = False
            if token not in ["NEWLINE", "INDENT", "DEDENT", "=", "+", "-", "*", "/"]:
                new_tokens[-1] = new_tokens[-1] + token
                cat = False
                continue
        if token in [".", "(", ")", "[", "]"]:
            if ti > 0 and new_tokens[-1] not in ["NEWLINE", "INDENT", "DEDENT", "=", "+", "-", "*", "/"]:
                # cat to the previous token
                new_tokens[-1] = new_tokens[-1] + token
                cat = True
            else:
                new_tokens.append(token)
                cat = True
            continue
        if token == ",":
            new_tokens[-1] = new_tokens[-1] + token
            continue
        new_tokens.append(token)
    tokens = new_tokens
    # import pdb; pdb.set_trace()

    while i < len(tokens):
        token = tokens[i]
        if token == "NEWLINE":
            code += "\n"
            for _ in range(indent_count):
                code += "\t"
        elif token == "INDENT":
            indent_count += 1
            code += "\t"
        elif token == "DEDENT":
            indent_count -= 1
            if code[-1] == "\t":
                code = code[:-1]
        else:
            code += token + " "
        i += 1
    lines = code.split("\n")
    taken_lines = []
    for line in lines:
        if len(line.strip()) > 0:
            taken_lines.append(line.rstrip())
    code = "\n".join(taken_lines)
    # code = code.replace(" . ", ".").replace(" .", ".").replace(". ", ".")
    # code = code.replace(" ,", ",")
    code = code.replace("NEWLINE", "\n")
    return code



processor_function = {
    "python": [PythonProcessor.for_to_while_random]
}

# need to reimplement first version for other languages!


class ForWhileTransformer(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "python": PythonProcessor.get_tokens
        }
        self.final_processor = processor_map[self.language]
        
    @func_set_timeout(5)
    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        ##### just for test #####
        # code = test_code[self.language]
        ##### just for test #####
        # mainly need to fix get_tokens_replace_while; get_tokens_replace_for
        oricode = code
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        if success:
            try:
                root_node = self.parse_code(
                    code=code
                )
                return_values = self.final_processor(
                    code=code.encode(),
                    root=root_node
                )
                if isinstance(return_values, tuple):
                    tokens, types = return_values
                else:
                    tokens, types = return_values, None
                code = " ".join(tokens)
                code = code.replace("\n", " NEWLINE ")
                code = beautify_python_code(code.split()).replace("\\", "")
            except:
                success=False
        return code, success


if __name__ == '__main__':
    input_map = {
        "python": ("python", python_code)
    }
   
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    # for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby", "cs"]:
    for lang in ["python"]:
        lang, code = input_map[lang]
        for_while_transformer = ForWhileTransformer(parser_path, lang)
        print(lang)
        print(code)
        print("=" * 100)
        code, meta = for_while_transformer.transform_code(code)
        print(code)
        print(meta)
        print("=" * 150)