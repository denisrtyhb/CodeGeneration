import ast
import warnings
import fnmatch
import os
from tqdm import tqdm

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
            "<SourceCode>",
            "Description",
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