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
    "python": [PythonProcessor.operator_bugs]
}


class OperatorBugs(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(OperatorBugs, self).__init__(parser_path=parser_path, language=language)
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
    from data_preprocessors.language_processors.utils import dfs_print
    python_code = "import sys\ninput = sys.stdin.readline\n\nt = int(input())\nfor _ in range(t):\n\tn = int(input())\n\ta = list(map(int, input().split()))\n\tans = [a[0]]\n\tfor i in range(1, n):\n\t\tif i % 2 == 0:\n\t\t\tif a[i] < a[i-1]:\n\t\t\t\tans.append(a[i])\n\t\telse:\n\t\t\tif a[i] > a[i-1]:\n\t\t\t\tans.append(a[i])\n\tprint(len(ans))\n\tprint(*ans)\n"
    input_map = {
        "python": ("python", python_code)
    }
    parser_path = "/home2/MyEXp/ranker_debug/parser/languages.so"
    for lang in ["python"]: 
        lang, code = input_map[lang]
        operandswap = OperatorBugs(
            parser_path, lang
        )
        print(lang)
        print(code)
        code, meta = operandswap.transform_code(code)
        print("============================================================================")
        print(code)
        print(meta)
        print("=" * 150)
        

