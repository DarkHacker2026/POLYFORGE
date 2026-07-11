import pycparser
from pycparser import c_ast
import sys
import io

class ExprEvaluator:
    def __init__(self, env):
        self.env = env
        
    def evaluate(self, node):
        if isinstance(node, c_ast.BinaryOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if node.op == '+': return left + right
            elif node.op == '-': return left - right
            elif node.op == '*': return left * right
            elif node.op == '/': return left / right
            elif node.op == '<': return left < right
            elif node.op == '>': return left > right
            elif node.op == '<=': return left <= right
            elif node.op == '>=': return left >= right
            elif node.op == '==': return left == right
            elif node.op == '!=': return left != right
            else:
                raise ValueError(f"Unknown BinaryOp: {node.op}")
        elif isinstance(node, c_ast.UnaryOp):
            if node.op == '-':
                return -self.evaluate(node.expr)
            elif node.op == '+':
                return +self.evaluate(node.expr)
            elif node.op == '!':
                return not self.evaluate(node.expr)
            else:
                raise ValueError(f"Unknown UnaryOp: {node.op}")
        elif isinstance(node, c_ast.Constant):
            val = node.value
            if val.lower().endswith('f'):
                val = val[:-1]
            if '.' in val:
                return float(val)
            else:
                return int(val)
        elif isinstance(node, c_ast.ID):
            return self.env[node.name]
        elif isinstance(node, c_ast.ArrayRef):
            array = self.evaluate(node.name)
            idx = self.evaluate(node.subscript)
            return array[idx]
        elif isinstance(node, c_ast.TernaryOp):
            cond = self.evaluate(node.cond)
            if cond:
                return self.evaluate(node.iftrue)
            else:
                return self.evaluate(node.iffalse)
        elif isinstance(node, c_ast.Cast):
            val = self.evaluate(node.expr)
            # node.to_type is Typename, .type is TypeDecl, .type is IdentifierType
            typ = node.to_type.type.type.names[0]
            if typ == 'float' or typ == 'double':
                return float(val)
            elif typ == 'int':
                return int(val)
            return val
        elif isinstance(node, c_ast.FuncCall):
            func_name = node.name.name
            args = [self.evaluate(arg) for arg in node.args.exprs] if node.args else []
            import math
            math_map = {
                'sinf': math.sin,
                'cosf': math.cos,
                'expf': math.exp,
                'sqrtf': math.sqrt,
                'fabsf': math.fabs,
                'powf': math.pow,
                'fminf': min,
                'fmaxf': max,
            }
            if func_name in math_map:
                return math_map[func_name](*args)
            else:
                raise ValueError(f"Unknown Math function: {func_name}")
        else:
            raise ValueError(f"Unknown AST node: {type(node)}")

def eval_cpp_expr(expr_str, env):
    src = f"void f() {{ float _result = {expr_str}; }}"
    parser = pycparser.CParser()
    try:
        ast = parser.parse(src)
    except pycparser.c_parser.ParseError as e:
        return f"ParseError: {e}", None
        
    try:
        func_def = ast.ext[0]
        compound = func_def.body
        decl = compound.block_items[0]
        init_expr = decl.init
        
        # Capture the AST output
        buf = io.StringIO()
        init_expr.show(buf=buf)
        ast_tree_str = buf.getvalue()
        
        evaluator = ExprEvaluator(env)
        res = evaluator.evaluate(init_expr)
        return res, ast_tree_str
    except Exception as e:
        return f"EvalError: {e}", None

def run_test(name, expr_str, env):
    print(f"====== Testing {name} ======")
    print(f"[RAW EXPRESSION]: {expr_str}")
    res, ast_tree = eval_cpp_expr(expr_str, env)
    if ast_tree:
        print(f"[AST TREE]:\n{ast_tree.strip()}")
    print(f"[COMPUTED RESULT]: {res}")
    print()

if __name__ == "__main__":
    env_vector_add = {'A': [1,2,3], 'B': [10,20,30], 'i': 0}
    env_ternary = {'g_idata': [5, 10, 15], 'i': 1, 'n': 2}
    env_cast = {'vecA': [100, 200], 'i': 1, 'finalValue': 10}
    env_math = {'x': 2.0, 'delta': 1.0}

    run_test('VectorAdd', 'A[i] + B[i] + 0.0f', env_vector_add)
    run_test('Ternary', '(i < n) ? g_idata[i] : 0', env_ternary)
    run_test('Cast', '(float)vecA[i] / finalValue', env_cast)
    run_test('Math', 'expf(-(x * x) / (2 * delta * delta))', env_math)
