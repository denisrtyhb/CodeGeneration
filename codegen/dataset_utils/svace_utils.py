import os
import subprocess
import json
_svace_checked = False

def check_svace_executable():
    """Checks if the 'svace' executable can be run from os.system.
    Only performs the check on first call.
    """
    global _svace_checked
    if _svace_checked:
        return
        
    try:
        # Use subprocess.run for a more reliable way to capture the exit code.
        result = subprocess.run(['svace', '--version'], capture_output=True, text=True)

        # Check the return code.  A return code of 0 usually indicates success.
        if result.returncode == 0:
            print ("SVACE version:", result.stdout)
        # The assertion passes if svace runs successfully (return code 0).
            pass
        else:
            raise AssertionError(f"'svace --version' command failed with return code {result.returncode}. Error output: {result.stderr}")

    except FileNotFoundError:
        raise AssertionError("'svace' executable not found in the system's PATH.")
    except Exception as e:
        raise AssertionError(f"An unexpected error occurred while checking 'svace': {e}")
    
    _svace_checked = True

def analyze_repo(path):
    
    check_svace_executable()
    command = f"\
    cd {path} && pwd && \
    svace init && echo 'Initiated svace' && \
    svace build --python . && echo 'Built svace' && \
    svace analyze --build-call-graph-only && echo 'Build call graph' \
    "
    os.system(command)

# path = '/home/jovyan/denis/CodeGeneration/data/example2'
import re

def is_number(text):
  pattern = r"^-?\d+(\.\d+)?$"  # Allows for optional negative sign and decimal part
  match = re.match(pattern, text)
  return bool(match)

def find_numbers(text):
    text = text[text.find('(')+1:]
    text = text[:text.find(')')]
    candidates = list(text.split("/"))
    if len(candidates) == 2 and is_number(candidates[0]) and is_number(candidates[1]):
        return list(map(int, candidates))
    return None

def parse_logs(path):
    project_name = os.path.split(path)[-1]
    log_path = os.path.join(path, f'.svace-dir/analyze-res/{project_name}.log')

    assert os.path.exists(log_path)

    with open(log_path, 'r') as file:
        logs = file.read()
    
    logs = logs[logs.find("Main interprocedural analysis phase..."):]



    log_lines = list(logs.split("\n"))
    i = 0
    while not log_lines[i].endswith("functions will be analyzed."):
        i += 1
    i += 1
    useful_logs = []
    logs_count = find_numbers(log_lines[i])[1]

    while len(useful_logs) < logs_count and i < len(log_lines):
        cur = find_numbers(log_lines[i])
        if cur is not None:
            useful_logs.append(log_lines[i])
        i += 1
    if len(useful_logs) != logs_count:
        print(f"COUNT MISMATCH. EXPECTED {logs_count} GOT {len(useful_logs)}")

    def get_name_and_path(line):
        line = line[line.find(")")+2:]
        first, last = line.split('(from ', 1)
        name = first.strip()

        # fixing things like '\x1b[0m_get_coords'
        if "tityList" in name:
            print([name], '->', [name[4:]])
        name = name[4:] # 4 is not a typo!!! \x1b[0m _get_coords

        full_path = last[:last.find(')')]
        rel_path = full_path[full_path.find(path)+len(path)+1:]
        return [name, rel_path]
    return list(map(get_name_and_path, useful_logs))


def not_init(vertice):
    i = vertice.rfind(":")
    last = vertice[i+1:]
    return not vertice.startswith("$$") and not vertice.endswith("$$")

def not_lambda(vertice):
    i = vertice.rfind(":")
    last = vertice[i+1:]
    return last != "<lambda>"

def not_module(vertice):
    i = vertice.rfind(":")
    last = vertice[i+1:]
    return last != "<module>"


def not_2(vertice):
    i = vertice.rfind(":")
    last = vertice[i+1:]
    return last[0] < '0' or last[0] > '9'

def transform_a_bit(function):
    if ('0' <= function[-1] and function[-1] <= '9') and function[-3:-1] == "::":
        return function[:-3]
    return function

def not_garbage(function):
    func_name = list(function.split(':'))[-1]
    if not_init(func_name) and not_lambda(func_name) and \
        not_module(func_name) and not_2(func_name) and \
            not function.startswith('builtin.py'):
        return True
    return False

def get_svace_graph(path):
    project_name = os.path.split(path)[-1]
    results_path = os.path.join(path, f'.svace-dir/analyze-res/call-graph/{project_name}-graph-order.json')

    if os.path.exists(results_path):
        print("Loading cache")
    else:
        analyze_repo(path)

    with open(results_path, 'r') as file:
        call_graph = json.load(file)['call-graph-v1.0']

    logs = parse_logs(path)

    name2path = dict(logs)


    final_graph = []

    for entry in call_graph:
        function = transform_a_bit(entry['function'])

        func_name = list(function.split(':'))[-1]
        if not_garbage(function):
            if func_name not in name2path:
                    print("UNEXPECTED CASE, not supposed to happen")
                    print(function, func_name)
            else:
                entry['path'] = name2path[func_name]
                entry['callees'] = list(filter(
                    not_garbage, 
                    map(transform_a_bit, entry['callees'])
                ))
                final_graph.append(entry)
            
    return final_graph

def print_stats(graph):
    vertices = len(graph)
    edges = sum(map(lambda x: len(x['callees']), graph))
    print(f"Vertices: {vertices}, Edges: {edges}")

if __name__ == "__main__":
    path = '/home/jovyan/denis/CodeGeneration/big_data/dspy-ai-2.0.4'
    call_graph = get_svace_graph(path)
    print_stats(call_graph)
