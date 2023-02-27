import copy
import os
import re
from typing import Union, Tuple

import numpy as np
from func_timeout import func_set_timeout
    
from data_preprocessors.language_processors import PythonProcessor
from data_preprocessors.transformations.transformation_base import TransformationBase
import io

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
    "python": [PythonProcessor.input_bugs]
}


class InputBugs(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(InputBugs, self).__init__(parser_path=parser_path, language=language)
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
            if success:
                try:
                    modified_code = modified_code.replace("\n", " NEWLINE ")
                    code = beautify_python_code(modified_code.split()).replace("\\", "")
                except:
                    success = False
        return code,success


if __name__ == '__main__':
    from data_preprocessors.language_processors.utils import dfs_print
    python_code = "import sys\ninput = sys.stdin.readline\n\nt = int(input())\nfor _ in range(t):\n\tn = int(input())\n\ta = list(map(int, input().split()))\n\tans = [a[0]]\n\tfor i in range(1, n):\n\t\tif i % 2 == 0:\n\t\t\tif a[i] < a[i-1]:\n\t\t\t\tans.append(a[i])\n\t\telse:\n\t\t\tif a[i] > a[i-1]:\n\t\t\t\tans.append(a[i])\n\tprint(len(ans))\n\tprint(*ans)\n"
    input_map = {
        "python": ("python", python_code)
    }
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    for lang in ["python"]: 
        lang, code = input_map[lang]
        operandswap = InputBugs(
            parser_path, lang
        )
        print(lang)
        print(code)
        code, meta = operandswap.transform_code(code)
        print("============================================================================")
        print(code)
        print(meta)
        print("=" * 150)
        

