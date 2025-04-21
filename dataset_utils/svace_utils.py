import os
import subprocess
import json

def check_svace_executable():
    """Checks if the 'svace' executable can be run from os.system.
    """
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

check_svace_executable()



def analyze_repo(path):
    command = f"\
    cd {path} && pwd && \
    svace init && echo 'Initiated svace' && \
    svace build --python . && echo 'Built svace' && \
    svace analyze --build-call-graph-only && echo 'Build call graph' \
    "
    os.system(command)

# path = '/home/jovyan/denis/CodeGeneration/data/example2'

def get_svace_graph(path):
    project_name = os.path.split(path)[-1]
    results_path = os.path.join(path, f'.svace-dir/analyze-res/call-graph/{project_name}-graph-order.json')

    if os.path.exists(results_path):
        print("Loading cache")
    else:
        analyze_repo(path)

    with open(results_path, 'r') as file:
        call_graph = json.load(file)['call-graph-v1.0']

    def change_name_to_readable(name):
        i = name.find('.')
        name = name[i+1:]
        return name

    # for i in call_graph:
    #     i['function'] = change_name_to_readable(i['function'])
    #     i['callees'] = list(map(change_name_to_readable, i['callees']))
    #     i.pop('source')

    def check_not_builtin(name):
        res = "builtin.py" not in name[:name.find(':')]
        return res

    call_graph = list(filter(
        lambda x: check_not_builtin(x['function']),
        call_graph
    ))

    for i in call_graph:
        i['callees'] = list(filter(check_not_builtin, i['callees']))

    return call_graph

def print_stats(graph):
    vertices = len(graph)
    edges = sum(map(lambda x: len(x['callees']), graph))
    print(f"Vertices: {vertices}, Edges: {edges}")

if __name__ == "__main__":
    path = '/home/jovyan/denis/CodeGeneration/big_data/dspy-ai-2.0.4'
    call_graph = get_svace_graph(path)
    print_stats(call_graph)
