import os
import clang.cindex
from clang.cindex import Index, CursorKind
import math


def _configure_libclang() -> None:
    candidates = []
    env_file = os.environ.get("LIBCLANG_PATH")
    if env_file:
        candidates.append(env_file)
    env_dir = os.environ.get("LIBCLANG_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, "libclang.so"))
    candidates.extend([
        "/usr/lib/llvm-21/lib/libclang.so",
        "/usr/lib/llvm-20/lib/libclang.so",
        "/usr/lib/llvm-19/lib/libclang.so",
        "/usr/lib/llvm-18/lib/libclang.so",
        "/usr/lib/x86_64-linux-gnu/libclang-21.so",
        "/usr/lib/x86_64-linux-gnu/libclang-20.so",
        "/usr/lib/x86_64-linux-gnu/libclang-19.so",
        "/usr/lib/x86_64-linux-gnu/libclang-18.so",
    ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            clang.cindex.Config.set_library_file(candidate)
            return


_configure_libclang()

class ClangExprEvaluator:
    def __init__(self, env):
        self.env = env
        
    def evaluate(self, node):
        kind = node.kind
        
        if kind == CursorKind.UNEXPOSED_EXPR or kind == CursorKind.PAREN_EXPR:
            children = list(node.get_children())
            if len(children) == 1:
                return self.evaluate(children[0])
            elif len(children) == 0:
                # might be a macro or something, just return spelling if exists
                return self.env.get(node.spelling, 0)

        elif kind == CursorKind.INTEGER_LITERAL:
            val = list(node.get_tokens())[0].spelling
            val = val.lower().rstrip('ul')
            return int(val, 0)
            
        elif kind == CursorKind.FLOATING_LITERAL:
            val = list(node.get_tokens())[0].spelling
            if val.lower().endswith('f'):
                val = val[:-1]
            return float(val)
            
        elif kind == CursorKind.DECL_REF_EXPR:
            return self.env.get(node.spelling, 0)
            
        elif kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(node.get_children())
            arr = self.evaluate(children[0])
            idx = self.evaluate(children[1])
            try:
                return arr[idx]
            except (IndexError, TypeError):
                return 0
                
        elif kind == CursorKind.BINARY_OPERATOR:
            children = list(node.get_children())
            left = self.evaluate(children[0])
            right = self.evaluate(children[1])
            
            # find operator
            left_end = children[0].extent.end.column
            op = None
            for t in node.get_tokens():
                if t.extent.start.column >= left_end:
                    op = t.spelling
                    break
            
            if op == '+': return left + right
            elif op == '-': return left - right
            elif op == '*': return left * right
            elif op == '/': return left / right if right != 0 else 0
            elif op == '%': return left % right if right != 0 else 0
            elif op == '<': return left < right
            elif op == '>': return left > right
            elif op == '<=': return left <= right
            elif op == '>=': return left >= right
            elif op == '==': return left == right
            elif op == '!=': return left != right
            else:
                raise ValueError(f"Unknown BinaryOp: {op}")
                
        elif kind == CursorKind.UNARY_OPERATOR:
            children = list(node.get_children())
            expr = self.evaluate(children[0])
            op = list(node.get_tokens())[0].spelling
            if op == '-': return -expr
            elif op == '+': return +expr
            elif op == '!': return not expr
            elif op == '~': return ~expr
            else:
                raise ValueError(f"Unknown UnaryOp: {op}")
                
        elif kind == CursorKind.CONDITIONAL_OPERATOR:
            children = list(node.get_children())
            cond = self.evaluate(children[0])
            if cond:
                return self.evaluate(children[1])
            else:
                return self.evaluate(children[2])
                
        elif kind == CursorKind.CSTYLE_CAST_EXPR or kind == CursorKind.CXX_FUNCTIONAL_CAST_EXPR:
            children = list(node.get_children())
            # children[0] might be TYPE_REF, last child is the expr
            expr = self.evaluate(children[-1])
            # get target type
            typ = node.type.spelling
            if 'float' in typ or 'double' in typ or '__half' in typ:
                return float(expr)
            elif 'int' in typ or 'short' in typ or 'long' in typ or 'size_t' in typ:
                return int(expr)
            return expr
            
        elif kind == CursorKind.CALL_EXPR:
            children = list(node.get_children())
            func_name = children[0].spelling
            args = [self.evaluate(c) for c in children[1:]]
            
            math_map = {
                'exp': math.exp, 'expf': math.exp,
                'sin': math.sin, 'sinf': math.sin,
                'cos': math.cos, 'cosf': math.cos,
                'pow': math.pow, 'powf': math.pow,
                'sqrt': math.sqrt, 'sqrtf': math.sqrt,
                'fmin': min, 'fminf': min,
                'fmax': max, 'fmaxf': max,
            }
            if func_name in math_map:
                return math_map[func_name](*args)
            else:
                # Might be a constructor call or custom func, just return the first arg
                return args[0] if args else 0
                
        elif kind == CursorKind.INIT_LIST_EXPR:
            children = list(node.get_children())
            if len(children) == 1:
                return self.evaluate(children[0])
            return [self.evaluate(c) for c in children]
            
        else:
            # Fallback
            children = list(node.get_children())
            if len(children) == 1:
                return self.evaluate(children[0])
            return 0


def evaluate_clang_ast(expr_str: str, env: dict):
    decls = ""
    for k, v in env.items():
        if k == 'math': continue
        if isinstance(v, (list, bytearray)):
            decls += f"float {k}[10000];\n"
        elif isinstance(v, float):
            decls += f"float {k} = 0;\n"
        else:
            decls += f"int {k} = 0;\n"
            
    src = f'''
    typedef float __half;
    float expf(float); float sinf(float); float cosf(float); float powf(float, float); float sqrtf(float);
    {decls}
    void f() {{
        auto _result = {expr_str};
    }}
    '''
    index = Index.create()
    tu = index.parse('test.cpp', args=['-std=c++11'], unsaved_files=[('test.cpp', src)])
    
    # Check for fatal parsing errors
    for diag in tu.diagnostics:
        if diag.severity >= 3: # Error or Fatal
            raise ValueError(f"Clang ParseError: {diag.spelling}")
            
    # Find the _result variable declaration
    for node in tu.cursor.walk_preorder():
        if node.kind == CursorKind.VAR_DECL and node.spelling == '_result':
            # The init expression is the last child
            init_expr = list(node.get_children())[-1]
            evaluator = ClangExprEvaluator(env)
            return evaluator.evaluate(init_expr)
            
    raise ValueError("Could not find AST node for expression")


if __name__ == '__main__':
    # Test tileRope failing expression
    expr = '__half{float(d % 11) / 10.0f - 0.5f}'
    env = {'d': 15}
    result = evaluate_clang_ast(expr, env)
    print(f"EVAL tileRope.cu expression '{expr}' with d=15:")
    print(f"Result = {result}")
