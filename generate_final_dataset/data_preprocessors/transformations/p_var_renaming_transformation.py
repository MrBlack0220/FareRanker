import math
import random
import re
from typing import Union, Tuple
import os
from data_preprocessors.language_processors import (
    PythonProcessor,
)
from func_timeout import func_set_timeout
from data_preprocessors.transformations.transformation_base import TransformationBase
import os
import string
import copy
import time
import random
python_code = "import sys\ninput = sys.stdin.readline\n\nt = int(input())\nfor _ in range(t):\n\tn = int(input())\n\ta = list(map(int, input().split()))\n\tans = [a[0]]\n\tfor i in range(1, n):\n\t\tif i % 2 == 0:\n\t\t\tif a[i] < a[i-1]:\n\t\t\t\tans.append(a[i])\n\t\telse:\n\t\t\tif a[i] > a[i-1]:\n\t\t\t\tans.append(a[i])\n\tprint(len(ans))\n\tprint(*ans)\n"

# python_code = """
# import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n
# """
input_map = {
    "python": ("python", python_code),
}
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

test_code = {"python": python_code}

processor_function = {
    "python": PythonProcessor,
}

tokenizer_function = {
    "python": PythonProcessor.get_tokens
}


class VarRenamerBase(TransformationBase):
    """ Base class for renaming variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerBase, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerBase"
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.TYPE_VARS = ["float", "list", "List", "int", "bool", "tuple", "str", "dict", "True", "False", "self", "return", "$", "in"]
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement","keyword_argument","decorator","ERROR"]
        self.package = ['combinations', 'math', 'floor', 'gcd', 'Counter', 'numpy', 'product', 'sqrt', 'ChainMap', 'functools', 'itertools', 'OrderedDict', 'tan', 'collections', 'Tuple', 'deque', 'time', 'permutations', 'accumulate', 'ceil', 'List', 'cos', 'exp', 'np', 'fabs', 'defaultdict', 'sin', 'lru_cache', 'heapq', 'typing', 'log2', 'fractions', 'sys', 'log', 'random']
        self.keywords = ['False','None', 'True','and','as', 'assert','break','class','continue', 'def','del','elif', 'else','except','finally', 'for', 'from','global','if','import','in','is','lambda', 'nonlocal','not','or','pass','raise', 'return','try','while','with','yield']
        
    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_not_var_ptype_var_names(self, root, code_string):
        var_names = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.parent.type) in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_import_var_names(self, root, code_string):
        # import_code_string = str(code_string)
        lines = code_string.split("\n")
        new_lines = []
        for line in lines:
            new_line = str(line)
            if "import" not in line:
                continue
            new_lines.append(new_line)
        import_code_string = "\n".join(new_lines)
        import_var_names = self.extract_var_names(self.parse_code(import_code_string), import_code_string)
        return list(set(import_var_names+self.package))

    def get_attribute_funcname(self,root,code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.prev_sibling == None:
                pass
            else:
                if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.prev_sibling.type)=='.':
                    var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return list(set(var_names))  

    def select_most_frequent_var(self, original_code, var_names):
        counts = {}
        for var in var_names:
            counts[var] = original_code.count(var)
        max_cnt = -1
        max_var = None
        for var in var_names:
            if counts[var] > max_cnt:
                max_cnt = counts[var]
                max_var = var
        return max_var

    def check_valid_var(self, var_name):
        if var_name == "":
            return False
        if var_name in self.TYPE_VARS:
            return False
        return var_name[0].isalpha() and var_name.replace("_", "").isalnum()

    def var_renaming(self, code_string, method, debug=False):
        
        try:
            # code_string = '"""User permissions."""\n\n# Django REST Framework\nfrom rest_framework.permissions import BasePermission\n\n\nclass IsAccountOwner(BasePermission):\n    """Allow access only to objects owned by the requesting user."""\n\n    def has_permission(self, request, view):\n        """Let object permission grant access."""\n        obj = view.get_object()\n        return self.has_object_permission(request, view, obj)\n        \n    # gga\n    # gga\n    def has_object_permission(self, request, view, obj):\n        """Check obj and user are the same. """\n        # happy\n        for i in range(10):\n            print("happy")\n        return request.user == obj'
            root = self.parse_code(code_string)
            original_code = self.tokenizer_function(code_string, root)
            var_names = self.extract_var_names(root, code_string)
            var_names = list(set(var_names))
            # for variables in import line, remove
            import_var_names = self.get_import_var_names(root, code_string)
            for ivn in import_var_names:
                if ivn in var_names: # possibly import in comments
                    var_names.remove(ivn)
            # for variables in type variables, remove
            for type_var in self.TYPE_VARS:
                if type_var in var_names:
                    var_names.remove(type_var)
            # for variable name appears in function/class name, remove
            not_var_ptype_var_names = self.get_not_var_ptype_var_names(root, code_string)
            for nvpvn in not_var_ptype_var_names:
                if nvpvn in var_names:
                    var_names.remove(nvpvn)
                    
            attribute_funcname = self.get_attribute_funcname(root, code_string)
            for funcname in attribute_funcname:
                if funcname in var_names:
                    var_names.remove(funcname)
            # selected_var is None if no suitable variable to replace
            
            for keyword in self.keywords:
                if keyword in var_names:
                    var_names.remove(keyword)
            if debug: import pdb; pdb.set_trace()

            if len(var_names) != 0:
                # we have suitable var to replace if selected_var is not None
                method = random.choice(["naive","alpha-numeric"])
                if method == "naive":
                    var_map = {}
                    if len(var_names)>1:
                        
                        random.shuffle(var_names)
                        num_to_rename = math.ceil(0.4 * len(var_names)) + 1 
                        var_names_to_rename = var_names[:num_to_rename]  
                        for idx, v in enumerate(var_names_to_rename):  
                            if idx != len(var_names_to_rename)-1:
                                var_map[v]= var_names_to_rename[idx+1]
                            else:
                                var_map[v]=var_names_to_rename[0]
                    else:
                        var_names_to_rename = var_names
                        while True:
                            c = random.choice(string.ascii_uppercase + string.ascii_lowercase)
                            if var_names_to_rename[0] != c :
                                var_map[var_names_to_rename[0]]= c  
                                break 
                            
                elif method == "alpha-numeric":
                    # random variable names, half alphabetics half numbers
                    num_to_rename = math.ceil(0.3 * len(var_names))
                    random.shuffle(var_names)
                    var_names_to_rename = var_names[:num_to_rename]
                    var_map = {}
                    for idx, v in enumerate(var_names_to_rename):
                        if len(v) == 1:
                            while True:
                                c=random.choice(string.ascii_uppercase + string.ascii_lowercase)
                                if c not in var_names:
                                    var_map[v] = c
                                    break
                        else:
                            while True:
                                c = random.choice(string.ascii_uppercase + string.ascii_lowercase) + \
                                        ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + \
                                        string.digits * 6) for _ in range(len(v) - 1))
                                if c not in var_names and c not in self.keywords:
                                    var_map[v] = c
                                    break

                modified_code = []
                for t in original_code:
                    if t in var_names_to_rename:
                        modified_code.append(var_map[t])
                    else:
                        modified_code.append(t)
            else:
                modified_code = original_code
            modified_code_string = " ".join(modified_code)
            return modified_code_string, modified_code != original_code
        
        except:
            return code_string,False
        
    @func_set_timeout(5)
    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        oricode = code
        code, success = self.var_renaming(code, method="naive")
        
        if success:
            try:
                code = code.replace("\n", " NEWLINE ")
                code = beautify_python_code(code.split()).replace("\\", "")
            except:
                success = False
        return code,success




if __name__ == '__main__':
    input_map = {
        "python": ("python", python_code),
    }
    
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    for lang in ["python"]:
        lang, code = input_map[lang]
        var_renamer = VarRenamerBase(
            parser_path, lang
        )
        # print("method:RN")
        print(code)
        print(40*"-")
        code1, meta = var_renamer.transform_code(code)
        print(code1)
        print(meta)
        print("=" * 100)
        
        
        var_renamer = VarRenamerBase(
            parser_path, lang
        )
        # print("method:Naive")
        print(code)
        print(40*"-")
        code2, meta = var_renamer.transform_code(code)
        print(code2)
        print(meta)
        print("=" * 100)
