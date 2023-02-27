
import math
import random
import re
from typing import Union, Tuple
import os
import copy
import string
from data_preprocessors.language_processors import (
    PythonProcessor
)
from func_timeout import func_set_timeout
from data_preprocessors.language_processors.utils import get_tokens
from data_preprocessors.transformations.transformation_base import TransformationBase
import os

processor_function = {
    "python": PythonProcessor
}

tokenizer_function = {
    "python": PythonProcessor.get_tokens
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


class VarmodifyBugs(TransformationBase):
    """ Base class for renaming variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarmodifyBugs, self).__init__(
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

    def var_modifying(self, code_string):
        # code_string = '"""User permissions."""\n\n# Django REST Framework\nfrom rest_framework.permissions import BasePermission\n\n\nclass IsAccountOwner(BasePermission):\n    """Allow access only to objects owned by the requesting user."""\n\n    def has_permission(self, request, view):\n        """Let object permission grant access."""\n        obj = view.get_object()\n        return self.has_object_permission(request, view, obj)\n        \n    # gga\n    # gga\n    def has_object_permission(self, request, view, obj):\n        """Check obj and user are the same. """\n        # happy\n        for i in range(10):\n            print("happy")\n        return request.user == obj'
        modify_code = None
        try:
            root = self.parse_code(code_string)
        except:
            return code_string,False
        try:
            original_code = self.tokenizer_function(code_string, root)
            var_names_ = self.extract_var_names(root, code_string)
            var_names = list(set(var_names_))
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
                    
                    
            if len(var_names)>=2:
                replace_token = random.choice(var_names)
                pos = []
                for idx,t in enumerate(original_code):
                    if t in var_names and t != replace_token:
                        pos.append(idx)
                modified_pos = random.choice(pos)
                modify_code = copy.deepcopy(original_code)
                modify_code[modified_pos] = replace_token
                modified_code_string = " ".join(modify_code)
                return modified_code_string,modify_code != replace_token
            elif len(var_names)==1 and len(var_names_)> 1:
                pos = []
                for idx,t in enumerate(original_code):
                    if t in var_names:
                        pos.append(idx)
                modified_pos = random.choice(pos)
                for iii in range(72):
                    replace_token = random.choice(string.ascii_uppercase + string.ascii_lowercase)
                    if replace_token != var_names[0]:
                        break
                modify_code = copy.deepcopy(original_code)
                modify_code[modified_pos] = replace_token
                modified_code_string = " ".join(modify_code)
                return modified_code_string,modify_code != replace_token
            else:
                return code_string,False
        except:
            code_string,False

    @func_set_timeout(5)
    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        code, success = self.var_modifying(code)
        
        if success:
            try:
                code = code.replace("\n", " NEWLINE ")
                code = beautify_python_code(code.split()).replace("\\", "")
            except:
                success = False
        return code,success


if __name__ == '__main__':
    python_code="import sys\ninput = sys.stdin.readline\n\nt = int(input())\nfor _ in range(t):\n\tn = int(input())\n\ta = list(map(int, input().split()))\n\tans = [a[0]]\n\tfor i in range(1, n):\n\t\tif i % 2 == 0:\n\t\t\tif a[i] < a[i-1]:\n\t\t\t\tans.append(a[i])\n\t\telse:\n\t\t\tif a[i] > a[i-1]:\n\t\t\t\tans.append(a[i])\n\tprint(len(ans))\n\tprint(*ans)\n"
    input_map = {
        "python": ("python", python_code)
    }
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    for lang in ["python"]:
        lang, code = input_map[lang]
        var_renamer = VarmodifyBugs(
            parser_path, lang
        )
        print(lang)
        print(code)
        print(40*"=")
        code, meta = var_renamer.transform_code(code)
        print(code)
        print(meta)
        print("=" * 150)