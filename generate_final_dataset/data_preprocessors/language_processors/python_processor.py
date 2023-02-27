'''
This file includes helper function for transformations on code structure. Many of the functions are built based on 
NatGEN (https://github.com/saikat107/NatGen) and Recoder https://github.com/amazon-science/recode
'''
from func_timeout import func_set_timeout
import numpy as np
import tokenize
from io import BytesIO
from tree_sitter import Node
from data_preprocessors.language_processors.utils import print_node, add_newline_token,count_lines,dfs_print,count_nodes,extract_statement_within_size
import random

class PythonProcessor:
    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        loop = f"NEWLINE for {control_variable} in range ( 0 ) : NEWLINE INDENT {body} NEWLINE DEDENT "
        return loop

    @classmethod
    def create_dead_while_loop(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"while False : NEWLINE INDENT {body} NEWLINE DEDENT"
        elif p < 0.66:
            return f"{control_variable} = 0 NEWLINE while {control_variable} < {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"
        else:
            return f"{control_variable} = 0 NEWLINE while {control_variable} > {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"

    @classmethod
    def create_dead_if(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"if False : NEWLINE INDENT {body} NEWLINE DEDENT"
        elif p < 0.66:
            return f"{control_variable} = 0 NEWLINE if {control_variable} < {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"
        else:
            return f"{control_variable} = 0 NEWLINE if {control_variable} > {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"


    @classmethod
    def get_tokens_insert_before(cls, code_str, root, insertion_code, insert_before_node, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        # if root == insert_before_node:
        #     tokens += insertion_code.split()
        if root.type == "comment":
            if include_comments: 
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code_str, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code_str, root)
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        if root == insert_before_node:
            tokens += insertion_code.split()
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code_str, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            ts = cls.get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        # print(" ".join(tokens).replace("NEWLINE", "\n"))
        return tokens

    @classmethod
    def get_tokens(cls, code, root, include_comments=True):
        # print(len(root.children) if root.children is not None else None, root.type, print_node(code, root))
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code, root)
                return tokens
            else:
                return [code[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        children = root.children
        if len(children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            ts = cls.get_tokens(code, child)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        # print(" ".join(tokens).replace("NEWLINE", "\n"))
        return tokens

    @classmethod
    def for_to_while_random(cls, code_string, parser):
        try:
            root = parser.parse_code(code_string)
        except:
            return None,code_string,False
        success = False
        try:
            loops = cls.extract_for_loops(root, code_string)
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        
            if not success:
                code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
            else:
                code_string = code_string.replace("\n", " NEWLINE ")
                code_string = cls.beautify_python_code(code_string.split())
        except:
            success = False
            pass
        return root, code_string, success

    ''' Start of Amazon addition '''
    @classmethod
    def for_to_while_first(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root, code_string)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = loops[0]
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
        else:
            code_string = cls.beautify_python_code(code_string.split())
        return root, code_string, success


    @classmethod
    def while_to_for_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_while_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = code_string.replace("\n", " NEWLINE ")
                code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
            else:
                code_string = cls.beautify_python_code(code_string.split())
        except:
            pass
        return root, code_string, False

    @classmethod
    def while_to_for_first(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_while_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = loops[0]
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
            else:
                code_string = cls.beautify_python_code(code_string.split())
        except:
            pass
        return root, code_string, False
    ''' End of Amazon addition '''

    @classmethod
    def extract_for_loops(cls, root, code_str):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'for_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def beautify_python_code(cls, tokens):
        indent_count = 0
        code = ""
        i = 0
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
        return code

    @classmethod
    def get_tokens_replace_for(cls, code_str, for_node, root, while_node, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments:
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code_str, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments:
                    tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code_str, root)
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code_str, root)
        for child in children:
            if child == for_node:
                tokens += while_node
            else:
                child_type = str(child.type)
                if child_type == "block":
                    tokens += ["NEWLINE", "INDENT"]
                tokens += cls.get_tokens_replace_for(code_str, for_node, child, while_node)
                if child_type.endswith("statement"):
                    tokens.append("NEWLINE")
                elif child_type == "block":
                    tokens.append("DEDENT")
        return tokens

    @classmethod
    def for_to_while(cls, code_string, root, fl, parser):
        if "range" in print_node(code_string, fl):
            try:
                identifier = fl.children[1]
                in_node = fl.children[2]
                range_node = fl.children[3]
                body_node = fl.children[5]
                range_function = range_node.children[0]
                range_function_name = cls.get_tokens(code_string, range_function)[0]
                stop_only = False
                if range_function_name == "range" \
                        and (str(identifier.type) == "identifier" and len(identifier.children) == 0) \
                        and (str(in_node.type) == "in" and len(in_node.children) == 0):
                    argument_list = range_node.children[1].children
                    args = []
                    for a in argument_list:
                        k = str(a.type)
                        if k not in ["(", ",", ")"]:
                            args.append(a)
                    start, stop, step = ["0"], ["0"], ["1"]
                    if len(args) == 1:
                        stop = cls.get_tokens(code_string, args[0])
                        stop_only = True
                    elif len(args) == 2:
                        start = cls.get_tokens(code_string, args[0])
                        stop = cls.get_tokens(code_string, args[1])
                    else:
                        start = cls.get_tokens(code_string, args[0])
                        stop = cls.get_tokens(code_string, args[1])
                        step = cls.get_tokens(code_string, args[2])
                    identifier_name = cls.get_tokens(code_string, identifier)[0]
                    if step[0] != "-":
                        while_stmt = [identifier_name, "="] + start + ["NEWLINE"] + \
                                    ["while", identifier_name, "<"] + stop + \
                                    [":", "NEWLINE", "INDENT"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", identifier_name, "+="] + step + \
                                    ["DEDENT", "NEWLINE"]
                    else:
                        while_stmt = [identifier_name, "="] + start + ["NEWLINE"] + \
                                    ["while", identifier_name, ">"] + stop + \
                                    [":", "NEWLINE", "INDENT"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", identifier_name, "-="] + step[1:] + \
                                    ["DEDENT", "NEWLINE"]
                    tokens = cls.get_tokens_replace_for(
                        code_str=code_string,
                        for_node=fl,
                        while_node=while_stmt,
                        root=root
                    )
                    code = cls.beautify_python_code(tokens)
                    return parser.parse_code(code), " ".join(tokens), True
            except:
                pass

        elif "in" in print_node(code_string, fl.children[2]):
            # for x in x_list...
            try:
                identifier = fl.children[1]
                in_node = fl.children[2]
                range_node = fl.children[3]
                body_node = fl.children[5]
                if (str(identifier.type) == "identifier" and len(identifier.children) == 0) \
                        and (str(in_node.type) == "in" and len(in_node.children) == 0):
                    identifier_name = cls.get_tokens(code_string, identifier)[0]
                    range_node_name = print_node(code_string, range_node)
                    invariance_name = "_"+identifier_name+"_i"
                    # put the invariance += 1 after loop body to make the problem harder
                    while_stmt =  [invariance_name, "=", "0"] + \
                                    ["while", invariance_name, "<", "len", "(", range_node_name, ")", ":", "NEWLINE", "INDENT"] + \
                                    [identifier_name, "=", range_node_name, "[", invariance_name, "]", "NEWLINE"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", invariance_name, "+="] + ["1"] + \
                                    ["DEDENT", "NEWLINE"]
                    tokens = cls.get_tokens_replace_for(
                        code_str=code_string,
                        for_node=fl,
                        while_node=while_stmt,
                        root=root
                    )
                    code = cls.beautify_python_code(tokens)
                    return parser.parse_code(code), " ".join(tokens), True
            except:
                pass
        return root, code_string, False

    @classmethod
    def extract_while_loops(cls, root):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'while_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def while_to_for(cls, code_string, root, wl, parser):
        # children = wl.children
        # condition = children[1]
        # body = children[2]
        # if str(condition.type) == 'parenthesized_expression':
        #     expr_tokens = get_tokens(code_string, condition.children[1])
        #     body_tokens = get_tokens(code_string, body)
        #     if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
        #         body_tokens = body_tokens[1:-1]
        #     tokens = cls.get_tokens_replace_while(
        #         code_str=code_string,
        #         while_node=wl,
        #         root=root,
        #         cond=expr_tokens,
        #         body=body_tokens
        #     )
        #     code = cls.beautify_python_code(tokens)
        #     return parser.parse_code(code), code, True
        return root, code_string, False

    @classmethod
    def get_tokens_replace_while(cls, code_str, while_node, root, cond, body):
        # if isinstance(code_str, str):
        #     code_str = code_str.encode()
        # assert isinstance(root, Node)
        # tokens = []
        # children = root.children
        # if len(children) == 0:
        #     tokens.append(code_str[root.start_byte:root.end_byte].decode())
        # for child in children:
        #     if child == while_node:
        #         tokens.extend(
        #             ["for", "(", ";"] + cond + [";", ")", "{"] + body + ["}"]
        #         )
        #     else:
        #         tokens += cls.get_tokens_replace_while(code_str, while_node, child, cond, body)
        # return tokens
        raise NotImplementedError

    @classmethod
    def extract_expression(self, root, code):
        expressions = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'comparison_operator':
                children_nodes = current_node.children
                keep = ["<", ">", "<=", ">=", "==", "!="]
                counter = 0
                for w in children_nodes:
                    if str(w.type) in keep:
                        counter = counter + 1
                if counter == 1:
                    expressions.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return expressions

    @classmethod
    def get_tokens_for_opswap(cls, code, root, left_oprd, operator, right_oprd, include_comments=True):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code, root)
            return tokens, None
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code, root)
                return tokens, None
            else:
                return [code[root.start_byte:root.end_byte].decode()], None
        if root.type == "decorator":
            # print(root.type, code[root.start_byte:root.end_byte].decode())
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens, None
        children = root.children
        if len(children) == 0:
            if root.start_byte == operator.start_byte and root.end_byte == operator.end_byte:
                opt = (code[operator.start_byte:operator.end_byte].decode())
                if opt == '<':
                    tokens.append('>')
                elif opt == '>':
                    tokens.append('<')
                elif opt == '>=':
                    tokens.append('<=')
                elif opt == '<=':
                    tokens.append('>=')
                elif opt == '==':
                    tokens.append('==')
                elif opt == '!=':
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
                tokens += add_newline_token(code, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            if child.start_byte == left_oprd.start_byte and child.end_byte == left_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, right_oprd, left_oprd, operator, right_oprd)
            elif child.start_byte == right_oprd.start_byte and child.end_byte == right_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, left_oprd, left_oprd, operator, right_oprd)
            else:
                ts, _ = cls.get_tokens_for_opswap(code, child, left_oprd, operator, right_oprd)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        return tokens, None

    @classmethod
    def operand_swap(cls, code_str, parser):
        code = code_str.encode()
        try:
            root = parser.parse_code(code)
        except:
            return code_str,False
        success = False
        try:
            expressions = cls.extract_expression(root, code)
            while not success and len(expressions) > 0:
                selected_exp = np.random.choice(expressions)
                expressions.remove(selected_exp)
                bin_exp = selected_exp
                condition = code[bin_exp.start_byte:bin_exp.end_byte].decode()
                bin_exp = bin_exp.children
                left_oprd = bin_exp[0]
                operator = bin_exp[1]
                right_oprd = bin_exp[2]
                try:
                    # code = code.replace("\n", " NEWLINE ")
                    code_list = cls.get_tokens_for_opswap(code, root, left_oprd, operator, right_oprd)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
            if not success:
                code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
            else:
                code_string = code_string.replace("\n", " NEWLINE ")
                code_string = cls.beautify_python_code(code_string.split())    
        except:
            success = False
            pass
        return code_string, success

    @classmethod
    def operand_swap_first(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        expressions = cls.extract_expression(root, code)
        success = False
        try:
            while not success and len(expressions) > 0:
                selected_exp = expressions[0]
                expressions.remove(selected_exp)
                bin_exp = selected_exp
                condition = code[bin_exp.start_byte:bin_exp.end_byte].decode()
                bin_exp = bin_exp.children
                left_oprd = bin_exp[0]
                operator = bin_exp[1]
                right_oprd = bin_exp[2]
                try:
                    # code = code.replace("\n", " NEWLINE ")
                    code_list = cls.get_tokens_for_opswap(code, root, left_oprd, operator, right_oprd)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
        else:
            code_string = code_string.replace("\n", " NEWLINE ")
            code_string = cls.beautify_python_code(code_string.split())
        return code_string, success

    @classmethod
    def extract_if_else(cls, root, code_str, operator_list):
        ext_opt_list = ["&&", "&", "||", "|"]
        expressions = []
        queue = [root]
        not_consider = []
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'if_statement':
                clause = code_str[current_node.start_byte:current_node.end_byte].decode()
                des = (current_node.children)[1]
                cond = code_str[des.start_byte:des.end_byte].decode()
                stack = [des]
                nodes = []
                while len(stack) > 0:
                    root1 = stack.pop()
                    if len(root1.children) == 0:
                        nodes.append(root1)
                    for child in root1.children:
                        stack.append(child)
                nodes.reverse()
                counter = 0
                extra_counter = 0
                for w in nodes:
                    if str(w.type) in operator_list:
                        counter = counter + 1
                    if str(w.type) in ext_opt_list:
                        extra_counter = extra_counter + 1
                if not (counter == 1 and extra_counter == 0):
                    continue
                children_nodes = current_node.children
                flagx = 0
                flagy = 0
                for w in children_nodes:
                    if str(w.type) == "else_clause":
                        flagx = 1
                    if str(w.type) == "elif_clause":
                        flagy = 1
                if flagx == 1 and flagy == 0:
                    expressions.append([current_node, des])
            for child in current_node.children:
                if child not in not_consider:
                    queue.append(child)

        return expressions

    @classmethod
    def get_tokens_for_blockswap(cls, code, root, first_block, opt_node, second_block, flagx, flagy):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens, None
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                return tokens, None
            else:
                return [code[root.start_byte:root.end_byte].decode()], None
        children = root.children
        if len(children) == 0:
            if root.start_byte == opt_node.start_byte and root.end_byte == opt_node.end_byte:
                op = code[root.start_byte:root.end_byte].decode()
                if op == "<":
                    tokens.append(">=")
                elif op == ">":
                    tokens.append('<=')
                elif op == ">=":
                    tokens.append('<')
                elif op == "<=":
                    tokens.append('>')
                elif op == "!=":
                    tokens.append('==')
                elif op == "==":
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            if child.start_byte == first_block.start_byte and child.end_byte == first_block.end_byte and flagx == 0 \
                    and str(
                child.type) == str(first_block.type):
                flagx = 1
                ts, _ = cls.get_tokens_for_blockswap(code, second_block, first_block, opt_node, second_block, flagx,
                                                     flagy)
            elif child.start_byte == second_block.start_byte and child.end_byte == second_block.end_byte and flagy == \
                    0 and str(
                child.type) == str(second_block.type):
                flagy = 1
                ts, _ = cls.get_tokens_for_blockswap(code, first_block, first_block, opt_node, second_block, flagx,
                                                     flagy)
            else:
                ts, _ = cls.get_tokens_for_blockswap(code, child, first_block, opt_node, second_block, flagx, flagy)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        return tokens, None

    @classmethod
    def block_swap(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        operator_list = ['<', '>', '<=', '>=', '==', '!=']
        pair = cls.extract_if_else(root, code, operator_list)
        success = False
        lst = list(range(0, len(pair)))
        try:
            while not success and len(lst) > 0:
                selected = np.random.choice(lst)
                lst.remove(selected)
                clause = pair[selected][0]
                des = pair[selected][1]
                st = [des]
                nodes = []
                while len(st) > 0:
                    root1 = st.pop()
                    if len(root1.children) == 0:
                        nodes.append(root1)
                        if (code[root1.start_byte:root1.end_byte].decode()) in operator_list:
                            opt_node = root1
                            break
                    for child in root1.children:
                        st.append(child)
                nodes = clause.children
                flag = 0
                for current_node in nodes:
                    if str(current_node.type) == 'block':
                        first_block = current_node
                    elif str(current_node.type) == 'else_clause':
                        new_list = current_node.children
                        for w in new_list:
                            if str(w.type) == 'block':
                                second_block = w
                                break
                flagx = 0
                flagy = 0
                try:
                    code_list = \
                        cls.get_tokens_for_blockswap(code, root, first_block, opt_node, second_block, flagx, flagy)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
        else:
            code_string = cls.beautify_python_code(code_string.split())
        return code_string, success


    @classmethod 
    def extract_priority(cls,root, code):
        priority = []   
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.type == "binary_operator": 
                for i in current_node.children:  
                    if i.type == "binary_operator" and current_node.children[1].type != i.children[1].type:
                        priority.append(current_node)
                        break
            if current_node.type == "comparison_operator":
                for i in current_node.children:  
                    if i.type == "binary_operator"\
                    or (i.type == "comparison_operator" and current_node.children[1].type != i.children[1].type):
                        priority.append(current_node)
                        break
            if current_node.type == "boolean_operator":
                for i in current_node.children:  
                    if i.type == "binary_operator"\
                    or i.type == "comparison_operator"\
                    or (i.type == "boolean_operator" and current_node.children[1].type != i.children[1].type):
                        priority.append(current_node)
                        break
            for child in current_node.children:
                queue.append(child)
        return priority
    

    @classmethod 
    def priority_trans(cls, code_str, parser):
        code = code_str.encode()
        try:
            root = parser.parse_code(code)
        except:
            return code_str,False
        modifycodestr = None
        success = False
        try:
            priority = cls.extract_priority(root, code) 
            while len(priority)>0 and not success:
                selected_pri = random.choice(priority)
                priority.remove(selected_pri)
                if selected_pri.children[0].type == "binary_operator"\
                    or selected_pri.children[0].type == "comparison_operator"\
                    or selected_pri.children[0].type == "boolean_operator": 
                    start = selected_pri.children[0].start_byte
                    end = selected_pri.children[0].end_byte
                    modifycodestr = code[:start].decode()+'('+code[start:end].decode()+')'+code[end:].decode()
                else:
                    start = selected_pri.children[2].start_byte
                    end = selected_pri.children[2].end_byte
                    modifycodestr = code[:start].decode()+'('+code[start:end].decode()+')'+code[end:].decode()
                success = True
        except:
            success = False
            modifycodestr = code_str
            pass
        return modifycodestr, success


    
    @classmethod
    def get_nodes(cls,root):
        l =[]
        l.append(root)
        for child in root.children:
            if child is not None:
                l += cls.get_nodes(child)
        return l
    
    
    @classmethod
    def operator_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        # TODO:
        try:
            operand = cls.extract_operator_bugs(root, code)
            while not success and len(operand) > 0:
                selected_oper = np.random.choice(operand)
                operand.remove(selected_oper)
                # try:
                if selected_oper.type == ">":
                    change = ">="
                elif selected_oper.type == ">=":
                    change = ">"
                elif selected_oper.type == "<":
                    change = "<="
                elif selected_oper.type == "<=":
                    change = "<"
                elif selected_oper.type == "//":
                    change = "/"
                elif selected_oper.type == "/":
                    change = "//"
                modifycodestr = code[:selected_oper.start_byte].decode()+ change + code[selected_oper.end_byte:].decode()
                success = True
        except:
            pass
        return modifycodestr, success

    @classmethod
    def extract_operator_bugs(cls,root, code):
        bugs_loction = ["<","<=",">",">=","/","//"]
        operand = []   
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.type in bugs_loction:   
                operand.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return operand
        
        
        
        
        

    @classmethod
    def index_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        try:
            subscripts = cls.extract_subscript(root, code)
            while not success and len(subscripts) > 0:
                selected_subs = random.choice(subscripts)
                subscripts.remove(selected_subs)
                try:
                    ans = str(eval(selected_subs.text.decode() + random.choice(["-","+"]) + str(random.randint(1,4))))
                except:
                    ans = selected_subs.text.decode() +random.choice(["-","+"]) + str(random.randint(1,4))
                modifycodestr = code[:selected_subs.start_byte].decode() + ans+ code[selected_subs.end_byte:].decode()
                success = True
        except:
            success = False
            pass
        return modifycodestr, success

    @classmethod
    def extract_subscript(self, root, code):
        
        subscript = []  
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'subscript':   
                if current_node.children[2].type !=  "slice":  
                    subscript.append(current_node.children[2])
                if current_node.children[2].type == "slice":
                    for child in current_node.children[2].children:
                        if child.type != ":":
                            subscript.append(child)
            for child in current_node.children:
                queue.append(child)
        return subscript
    
    
    @classmethod
    def functionargs_bugs(cls, code_str, parser):
        modifycodrstr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        try:
            functionargs = cls.extract_functionargs(root, code)  
            var_names = cls.extract_var_names_finally(root,code_str,parser)
            while not success and len(functionargs) > 0:
                selected_args = random.choice(functionargs)
                functionargs.remove(selected_args)
                args_list = []
                args_count = 0
                for child in selected_args.children: 
                    if child.type != "," and child.type != ")" and child.type != "(":
                        args_list.append(((child.start_byte,child.end_byte),child.text))
                        args_count+=1
                if args_count > 1: 
                    prob = random.uniform(0,1)
                    if prob < 0.33:
                            args_text= cls.delete_args(args_list,code)
                            modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                    elif prob < 0.66:
                            args_text= cls.change_args(args_list,code)
                            modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                    else: 
                            args_text= cls.add_args(args_list,var_names,code)
                            modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                elif args_count == 1: 
                    prob = random.uniform(0,1)
                    if prob < 0.5:
                        args_text= cls.delete_args(args_list,code)
                        modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                    else: 
                        args_text= cls.add_args(args_list,var_names,code)
                        modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                else:  
                    args_text= cls.add_args(args_list,var_names,code)
                    modifycodrstr = code[:selected_args.start_byte].decode() + args_text + code[selected_args.end_byte:].decode()
                success = True
        except:
            success = False
            pass
        return modifycodrstr, success




    @classmethod
    def extract_functionargs(cls, root, code):
        
        functionargs = []   
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'argument_list': 
                functionargs.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return functionargs
    

    @classmethod
    def extract_var_names_finally(cls,root,code_string,parser):
        TYPE_VARS = ["float", "list", "List", "int", "bool", "tuple", "str", "dict", "True", "False", "self", "return", "$", "in"]
        var_names = cls.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        import_var_names = cls.get_import_var_names(root,code_string,parser)
        for ivn in import_var_names:
            if ivn in var_names: # possibly import in comments
                var_names.remove(ivn)
        for type_var in TYPE_VARS:
            if type_var in var_names:
                var_names.remove(type_var)
        not_var_ptype_var_names = cls.get_not_var_ptype_var_names(root, code_string)
        for nvpvn in not_var_ptype_var_names:
            if nvpvn in var_names:
                var_names.remove(nvpvn)
        attribute_funcname = cls.get_attribute_funcname(root, code_string)
        for funcname in attribute_funcname:
            if funcname in var_names:
                var_names.remove(funcname)
        return var_names
    
    @classmethod
    def extract_var_names(cls, root, code_string):
        var_names = []
        queue = [root]
        not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement","keyword_argument","decorator","ERROR"]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in not_var_ptype:
                var_names.append(cls.get_tokens(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names
    
    @classmethod
    def get_import_var_names(self, root,code_string,parser):
        # import_code_string = str(code_string)
        package = ['combinations', 'math', 'floor', 'gcd', 'Counter', 'numpy', 'product', 'sqrt', 'ChainMap', 'functools', 'itertools', 'OrderedDict', 'tan', 'collections', 'Tuple', 'deque', 'time', 'permutations', 'accumulate', 'ceil', 'List', 'cos', 'exp', 'np', 'fabs', 'defaultdict', 'sin', 'lru_cache', 'heapq', 'typing', 'log2', 'fractions', 'sys', 'log', 'random']
        lines = code_string.split("\n")
        new_lines = []
        for line in lines:
            new_line = str(line)
            if "import" not in line:
                continue
            new_lines.append(new_line)
        import_code_string = "\n".join(new_lines)
        import_var_names = self.extract_var_names(parser.parse_code(import_code_string), import_code_string)
        return list(set(import_var_names+package))
    
    @classmethod
    def get_not_var_ptype_var_names(cls, root, code_string):
        var_names = []
        queue = [root]
        not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement","keyword_argument","decorator","ERROR"]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.parent.type) in not_var_ptype:
                var_names.append(cls.get_tokens(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    @classmethod
    def get_attribute_funcname(cls,root,code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.prev_sibling == None:
                pass
            else:
                if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.prev_sibling.type)=='.':
                    var_names.append(cls.get_tokens(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return list(set(var_names))  
    
    
    @classmethod
    def delete_args(cls,args_list,code):
        selected_node = random.choice(args_list)
        args_list.remove(selected_node)
        args_text_list= [i[1].decode() for i in args_list]
        args_text = "(" + ",".join(args_text_list) + ")"
        return args_text
    

    @classmethod
    def change_args(cls,args_list,code):
        random.shuffle(args_list)
        args_text_list= [i[1].decode() for i in args_list]
        args_text = "(" + ",".join(args_text_list) + ")"
        return args_text
    
    

    @classmethod
    def add_args(cls,args_list,var_names,code):
        selected_var = np.random.choice(var_names)
        args_text_list= [i[1].decode() for i in args_list]
        if len(args_text_list)>0:
            n = random.randint(0,len(args_text_list))
        else:
            n = 0
        args_text_list.insert(n,selected_var)  
        args_text = "(" + ",".join(args_text_list) + ")"

        return args_text
    
    @classmethod
    def range_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        try:
            ranges = cls.extract_ranges(root, code)  
            while not success and len(ranges) > 0:
                selected_ranges = random.choice(ranges)
                ranges.remove(selected_ranges)
                args_list = []
                for child in selected_ranges.children:  
                    # if child.type == "identifier" or child.type == "integer" or child.type=="call":
                    if child.type != "," and child.type !=")" and child.type !="(":
                        args_list.append(child)
                selected_args = random.choice(args_list)
                if selected_args.type == "integer":
                    n = random.randint(1,5)  
                    ans = eval(selected_args.text.decode() + random.choice(["-","+"]) + str(n))
                    modifycodestr = code[:selected_args.start_byte].decode() + str(ans) + code[selected_args.end_byte:].decode()
                else:
                    n = random.randint(1,5)  
                    modifycodestr = code[:selected_args.start_byte].decode()+selected_args.text.decode()+ random.choice(["-","+"]) + str(n) + code[selected_args.end_byte:].decode()
                success = True
        except:
            success = False
            pass
        return modifycodestr, success

    @classmethod
    def extract_ranges(self, root, code):
        ranges = []  
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'argument_list' and current_node.parent.type == "call"\
                and current_node.prev_sibling.text == b"range":    
                ranges.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return ranges



    @classmethod
    def boolean_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str , success
        
        try:
            boolean_operator = cls.extract_boolean_operator(root, code)  
            while not success and len(boolean_operator) > 0:
                # try:
                selected_boo = random.choice(boolean_operator)
                boolean_operator.remove(selected_boo)
                boos_list = []
                for child in selected_boo.children:
                    boos_list.append(child)
                p = random.uniform(0,1)
                if p < 0.5:  
                    boo_opr = boos_list[1]
                    if boo_opr.type == "and":
                        moditycode = code[:boo_opr.start_byte] + "or".encode() + code[boo_opr.end_byte:]
                    else:
                        moditycode = code[:boo_opr.start_byte] + "and".encode() + code[boo_opr.end_byte:]
                else: 
                    boo_opr = boos_list[1]  
                    boo_num = random.choice([boos_list[0],boos_list[2]])   
                    if boo_opr.start_byte < boo_num.start_byte:  
                        moditycode = code[:boo_opr.start_byte]+code[boo_num.end_byte:]
                    else: 
                        moditycode = code[:boo_num.start_byte]+code[boo_opr.end_byte:]
                modifycodestr = moditycode.decode()
                success = True
                # except:
                #     success = False
                #     continue
        except:
            success = False
            pass
        return modifycodestr, success
    

    @classmethod
    def extract_boolean_operator(cls,root, code):
        boolean_operator = [] 
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.type == 'boolean_operator':    
                boolean_operator.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return boolean_operator
    
    

    @classmethod
    def condition_bugs(cls, code_str, parser):
        modifycodestr = None
        success = False
        code = code_str.encode()
        try:
            root = parser.parse_code(code)
        except:
            return code_str, success 
        try:
            condition = cls.extract_condition(root, code)
            lines_limit = 3
            statement_size = []
            for statement in condition:
                statement_size.append(count_lines(statement))
            small_condition = [condition[idx] for idx,i in enumerate(statement_size) if i <= lines_limit]
            while not success and len(small_condition) > 0:
                selected_cond = random.choice(small_condition)
                small_condition.remove(selected_cond)
                modifycode = code[:selected_cond.start_byte]+code[selected_cond.end_byte:]
                modifycodestr = modifycode.decode()
                success = True
        except:
            success = False
            pass
        return modifycodestr, success 
    
    

    @classmethod       
    def extract_condition(cls,root, code):
        condition = []  
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if current_node.type == 'if_statement':
                if current_node.next_sibling==None:
                    condition.append(current_node)
                elif current_node.next_sibling.type != "elif_clause" or current_node.next_sibling.type!="else_clause":
                    condition.append(current_node)
            if current_node.type == "elif_clause" or current_node.type =="else_clause":
                condition.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return condition
    


    
    # TODO:
    @classmethod
    def input_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        try:
            original_node_count = count_nodes(root)
            max_node_in_statement = int(original_node_count / 2)
            statement_markers = None
            statements = extract_statement_within_size(
                root, max_node_in_statement, statement_markers,
                code_string=code_str, tokenizer=cls.get_tokens,
            )
            var_names = list(set(cls.extract_var_names_finally(root,code_str,parser)))
            while len(statements) > 0 and not success:
                insert_before = np.random.choice(statements)
                statements.remove(insert_before)
                input_code = cls.create_input_statement(var_names) +' NEWLINE'
                
                modifycodestr = " ".join(cls.get_tokens_insert_before(
                        code_str=code_str, root=root, insertion_code=input_code,
                        insert_before_node=insert_before
                    ))  

                success = True
        except:
            success = False
            pass
        return  modifycodestr, success
    

    @classmethod
    def create_input_statement(cls,var_name):
        p = np.random.uniform(0, 1)  
        s = np.random.uniform(0, 1)  
        try :
            if p < 0.25:
                var_1 = random.choice(var_name)
                if s < 0.2:
                    return f"{var_1} = float(input())"
                return f"{var_1} = int(input())"
            elif p < 0.50:
                var_1 = random.choice(var_name)
                if s < 0.2:
                    return f"{var_1} = list(map(float, input().split()))"
                return f"{var_1} = list(map(int, input().split()))"
            elif p < 0.75:
                var_1 = random.choice(var_name)
                return f"{var_1} = input()"
            else:
                try: 
                    num = random.randint(2,5)
                    vars = random.sample(var_name,num)
                    vars = ','.join(vars)
                    if s < 0.2:
                        return f"{vars} = map(float, input().split())"
                    return f"{vars} = map(int, input().split())"
                except:
                    try: 
                        num = random.randint(2,4)
                        vars = random.sample(var_name,num)
                        vars = ','.join(vars)
                        if s < 0.2:
                            return f"{vars} = map(float, input().split())"
                        return f"{vars} = map(int, input().split())"
                    except:
                        try:  
                            num = random.randint(2,3)
                            vars = random.sample(var_name,num)
                            vars = ','.join(vars)
                            if s < 0.2:
                                return f"{vars} = map(float, input().split())"
                            return f"{vars} = map(int, input().split())"
                        except:
                            try:  
                                num = 2
                                vars = random.sample(var_name,num)
                                vars = ','.join(vars)
                                if s < 0.2:
                                    return f"{vars} = map(float, input().split())"
                                return f"{vars} = map(int, input().split())"
                            except: 
                                vars = random.choice(var_name)
                                if s < 0.2:
                                    return f"{vars} = float(input())"
                                return f"{vars} = int(input())"
        except:
            return f"var = int(input())"
    

    

    @classmethod 
    def priority_bugs(cls, code_str, parser):
        modifycodestr = None
        code = code_str.encode()
        success = False
        try:
            root = parser.parse_code(code)
        except:
            return code_str,success
        try:
            priority = cls.extract_priority(root, code)  
            while not success and len(priority) > 0:

                selected_pri = random.choice(priority)
                priority.remove(selected_pri)
                if selected_pri.children[0].type == "binary_operator"\
                    or selected_pri.children[0].type == "comparison_operator"\
                    or selected_pri.children[0].type == "boolean_operator": 

                    start = selected_pri.children[0].children[2].start_byte
                    end = selected_pri.children[2].end_byte
                    modifycodestr = code[:start].decode()+'('+code[start:end].decode()+')'+code[end:].decode()
                    success = True
                else:
                    end = selected_pri.children[2].children[0].end_byte
                    start = selected_pri.children[0].start_byte
                    modifycodestr = code[:start].decode()+'('+code[start:end].decode()+')'+code[end:].decode()
                    success = True
        except:
            success = False
            pass
        return modifycodestr, success

def get_python_tokens(code, root=None):
    if isinstance(code, bytes):
        code = code.decode()
    tokens = []
    for token in tokenize.tokenize(BytesIO(code.encode("utf-8")).readline):
        if token.type == 0 or token.type >= 58:
            continue
        elif token.type == 4:
            tokens.append("NEWLINE")
        elif token.type == 5:
            tokens.append("INDENT")
        elif token.type == 6:
            tokens.append("DEDENT")
        else:
            tokens.append(token.string)
    return tokens, None
