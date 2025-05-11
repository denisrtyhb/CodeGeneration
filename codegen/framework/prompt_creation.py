general_prompt = """
You are an expert Python programming language developer;
You generate a new function in the .py file in the code repository;
You are given a function signature and a description, and you should generate the implementation of the function, reusing the code from the code repository whenever possible.
When using some other function, you must write import statement for that function.
Write only the implementation of the function, no other text.
{context_str}
Function signature:
{function_signature}
"""

def context_func_to_str(context_func):
    return f"{context_func['raw_source_code']}\n\n"

def create_prompt_with_context(signature, description, context):
    context_str = "Relevant code from the code repository:\n"
    for i, c in context:
        context_str += context_func_to_str(c)
    return general_prompt.format(context_str=context_str, function_signature=signature)

def create_prompt_without_context(signature, description):
    return general_prompt.format(context_str="", function_signature=signature)
