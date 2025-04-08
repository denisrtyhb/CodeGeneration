import ast
import warnings

class CallFinderVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
    def find_calls(self, node):
        assert isinstance(node, ast.FunctionDef), ""
        self.call_array = []
        self.visit(node)
        return self.call_array

    def visit_ClassDef(self, node):
        pass
        warnings.warn("Detected class definition inside function definition")
        # assert False, "Classes are not supposed to appear in function definitions"

    def visit_Call(self, node):
        self.call_array.append(get_full_attribute_name(node.func))
        
        self.generic_visit(node)

class GraphMakerVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.mini_visitor = CallFinderVisitor()
    def construct_graph(self, node):
        self.graph = []
        self.cur_class = None
        self.visit(node)
        return self.graph

    def visit_Call(self, node):
        pass

    def visit_ClassDef(self, node):
        self.class_graph = []
        self.generic_visit(node)

        # post_processing
        for name, edges, other_node in self.class_graph:
            self.graph.append((
                name,
                list(map(lambda x: x.replace("self", node.name), edges)),
                other_node
            ))
        self.cur_class = None

    def visit_FunctionDef(self, node):
        entry = (
            node.name if self.cur_class is None else f"{self.cur_class}.{node.name}",
            self.mini_visitor.find_calls(node),
            node
        )
        if self.cur_class:
            self.class_graph.append(entry)
        else:
            self.graph.append(entry)

def get_full_attribute_name(node):
    """
    Recursively constructs the full attribute name (e.g., module.submodule.f)
    from an ast.Attribute node.
    """

    if isinstance(node, ast.Name):
        return node.id
    elif not isinstance(node, ast.Attribute):

        warnings.warn("Detected other call type")
        return "<unknown>"

    parts = []
    current = node

    while isinstance(current, ast.Attribute):
        parts.insert(0, current.attr)  # Add attribute name to the beginning
        current = current.value # Move to the value

    if isinstance(current, ast.Name):
        parts.insert(0, current.id)  # Add the base name

    return ".".join(parts)

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

def get_function_declarations(tree):
    """
    Extracts all function definitions (declarations) from an AST.

    Args:
      tree: The AST to traverse.

    Returns:
      A list of ast.FunctionDef nodes representing function declarations.
    """
    function_defs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)
    return function_defs

def get_imported_function_names(tree):
    imported_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Handles 'from module import name1, name2, ...'
            module_name = node.module # The module imported from

            for alias in node.names:
                imported_functions.append(alias.name + "." + module_name)  # (function_name, module_name)

        # elif isinstance(node, ast.Import):
        #   # Handles 'import module as alias' or 'import module'
        #   for alias in node.names:
        #       # Look for usages of the imported module in calls
        #       module_name = alias.name
        #       module_alias = alias.asname if alias.asname else module_name #Alias if it exist, or module name

        #       for sub_node in ast.walk(tree):
        #           if isinstance(sub_node, ast.Call) and \
        #               isinstance(sub_node.func, ast.Attribute) and \
        #               isinstance(sub_node.func.value, ast.Name) and \
        #               sub_node.func.value.id == module_alias:
        #                   imported_functions.append((sub_node.func.attr, module_name))


    return imported_functions

def find_function_calls(function_body):
    """
    Finds all function calls within the body of a function (AST node).

    Args:
      function_body: A list of AST nodes representing the body of a function.

    Returns:
      A list of ast.Call nodes representing function calls within the function.
    """
    function_calls = []
    for node in function_body:  # Iterate over nodes in the function body
        for sub_node in ast.walk(node):  # Walk the subtree rooted at each node
            if isinstance(sub_node, ast.Call):
                function_calls.append(sub_node)
    return function_calls