import clang.cindex
from clang.cindex import Index, CursorKind

src = '''
typedef struct { unsigned short x; } __half;
float expf(float);
void f(int i, float* A, float* B, float n, int d, float theta) {
    float r1 = A[i] + B[i];
    float r2 = (i < n) ? A[i] : 0;
    float r3 = (float)A[i] / 10.0f;
    float r4 = expf(-(A[i] * B[i]));
    float r5 = __half{float(d % 11) / 10.0f - 0.5f};
}
'''
index = Index.create()
tu = index.parse('test.cpp', args=['-std=c++11'], unsaved_files=[('test.cpp', src)])

def print_ast(node, indent=''):
    tokens = [t.spelling for t in node.get_tokens()]
    print(f'{indent}{node.kind.name} "{node.spelling}" tokens={tokens}')
    for c in node.get_children():
        print_ast(c, indent + '  ')

print_ast(tu.cursor)
