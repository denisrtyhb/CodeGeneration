import ast
import astunparse
import warnings
import fnmatch
import os
from tqdm import tqdm

def get_function_comment(function_node):
    if function_node.body and function_node.body[0] and isinstance(function_node.body[0], ast.Expr) and isinstance(function_node.body[0].value, ast.Constant) and isinstance(function_node.body[0].value.value, str):
        return function_node.body[0].value.value

def get_function_code(function_node):
    # Create a copy of the function node so we don't modify the original AST
    function_copy = ast.copy_location(ast.FunctionDef(
        name=function_node.name,
        args=function_node.args,
        body=[], # Empty the body initially; we'll populate it selectively
        decorator_list=function_node.decorator_list,
        returns=function_node.returns,
        type_comment=function_node.type_comment
    ), function_node)

    # Iterate through the original function body and only add non-comment nodes to the copy
    # for node in function_node.body:
    #     if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)): #Skip docstrings
    #         function_copy.body.append(node)

    try:
        source_code = astunparse.unparse(function_copy)
        return source_code.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        print(f"Error unparsing function: {e}")
        return None

class GraphMakerVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
    def find_functions(self, node):
        self.functions = []
        self.cur_class = None
        self.visit(node)
        return self.functions

    def visit_Call(self, node):
        pass

    def visit_ClassDef(self, node):

        self.cur_class = node.name
        self.generic_visit(node)
        self.cur_class = None

    def visit_FunctionDef(self, node):
        entry = [
            node.name if self.cur_class is None else f"{self.cur_class}.{node.name}",
            get_function_code(node),
            get_function_comment(node),
        ]
        self.functions.append(entry)
def construct_ast_from_file(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()

        tree = ast.parse(content)
        return tree

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None
    except SyntaxError as e:
        print(f"Error: Syntax error in file: {filename}")
        print(e)  # Print the specific syntax error
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def process_file(path, verbose=False):
    ast_tree = construct_ast_from_file(path)
    assert ast_tree, f"Failed to construct AST for {path}"

    visitor = GraphMakerVisitor()

    vertices = visitor.find_functions(ast_tree)
    return vertices


basic_functions = {
    'set',
    'split',
    'min',
    'max',
    'int',
    'str',
    'zip',
    'len',
    'list',
    'tuple',
    'dict',
    'range',
    'enumerate',
    'filter',
    'map',
    'sorted',
    'sum',
    'any',
    'all',
    'round',
    'isinstance',
    'getattr',    
}

def get_vertices(path):

    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.py'):
            files.append(os.path.join(root, filename))

    all_vertices = []

    local_paths = {}
    with tqdm(total=len(files), desc="", leave=True) as pbar:
        for filepath in files:
            pbar.set_description(f"Processing {filepath}")
            
            assert filepath.endswith(".py")
            local_path = filepath[len(path)+1:-3]
            vertices = process_file(filepath)
            
            for entry in vertices:
                all_vertices.append([
                    f"{local_path}.{entry[0]}",
                    entry[1],
                    entry[2]
                ])
            pbar.update(1)
    
    return all_vertices