import re
from typing import Union, Tuple
import os
from func_timeout import func_set_timeout
import numpy as np
from sympy import jscode
import random
import time
from data_preprocessors.language_processors import (
    PythonProcessor,
)
from data_preprocessors.language_processors.utils import extract_statement_within_size, get_tokens, \
    get_tokens_insert_before, count_nodes, print_node,dfs_print
from data_preprocessors.transformations.transformation_base import TransformationBase
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



python_code = "import sys\ninput = sys.stdin.readline\n\nt = int(input())\nfor _ in range(t):\n\tn = int(input())\n\ta = list(map(int, input().split()))\n\tans = [a[0]]\n\tfor i in range(1, n):\n\t\tif i % 2 == 0:\n\t\t\tif a[i] < a[i-1]:\n\t\t\t\tans.append(a[i])\n\t\telse:\n\t\t\tif a[i] > a[i-1]:\n\t\t\t\tans.append(a[i])\n\tprint(len(ans))\n\tprint(*ans)\n"

test_code = {"python": python_code}

processor_function = {
    "python": PythonProcessor,
}

tokenizer_function = {
    "python": PythonProcessor.get_tokens,
}

insertion_function = {
    "python": PythonProcessor.get_tokens_insert_before,
}


class DeadCodeInserter(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(DeadCodeInserter, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        self.insertion_function = insertion_function[self.language]

    def insert_random_dead_code(self, code_string, max_node_in_statement=-1, pos_type="random"):
        try:
            root = self.parse_code(code_string)
        except:
            return code_string,False
        try:
            original_node_count = count_nodes(root)
            if max_node_in_statement == -1:
                max_node_in_statement = int(original_node_count / 2)
            if self.language == "ruby":
                statement_markers = ["assignment", "until", "call", "if", "for", "while"]
            else:
                statement_markers = None
            statements = extract_statement_within_size(
                root, max_node_in_statement, statement_markers,
                code_string=code_string, tokenizer=self.tokenizer_function,
            )
            original_code = " ".join(self.tokenizer_function(code_string, root))
            if pos_type == "random":
                np.random.shuffle(statements)
            dead_code_statements = list(statements) # dead_code_statements for random_stmt, statements for insert_before
            while len(dead_code_statements) > 0:
                random_stmt, insert_before = np.random.choice(dead_code_statements, 2)
                dead_code_statements.remove(random_stmt)
                dead_code_body = " ".join(self.tokenizer_function(code_string, random_stmt)).strip()
                dead_code_function = np.random.choice(
                    [
                        self.processor.create_dead_for_loop,
                        self.processor.create_dead_while_loop,
                        self.processor.create_dead_if
                    ]
                )
                dead_code = dead_code_function(dead_code_body)
                modified_code = " ".join(
                    self.insertion_function(
                        code_str=code_string, root=root, insertion_code=dead_code,
                        insert_before_node=insert_before
                    )
                )
                if modified_code != original_code:
                    
                    return modified_code, True
        except:
            pass
        return original_code, False  # modified_code

    @func_set_timeout(5)
    def transform_code(
            self,
            code: Union[str, bytes],
            first_half = False
    ) -> Tuple[str, object]:
        code, success = self.insert_random_dead_code(code, -1)
        
        if success:
            try:
                code = code.replace("\n", " NEWLINE ")
                code = beautify_python_code(code.split()).replace("\\", "")
            except:
                success=False
        return code,success



if __name__ == '__main__':
    input_map = {
        "python": ("python", python_code),
    }
    code_directory = ""
    # parser_path = os.path.join(code_directory, "parser/languages.so")
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    # for lang in ["c", "cpp", "java", "python", "php", "ruby", "js", "go", "cs"]:
    for lang in [ "python"]:
        lang, code = input_map[lang]
        print(code)
        print("=" * 100)
        dead_code_inserter = DeadCodeInserter(
            parser_path, lang
        )
        print(lang)
        code, meta = dead_code_inserter.transform_code(code)
        print(code)
        print(meta)
        
