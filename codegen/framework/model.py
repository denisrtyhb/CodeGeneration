
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
# Global device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model():
    model_name = "ibm-granite/granite-8B-code-base-128k"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Global variables for first load
MODEL, TOKENIZER = load_model()

def ask_model(prompt_text):

    # # Extract content from messages
    # prompt_text = ""
    # for message in prompt:
    #     if message["role"] == "system":
    #         prompt_text += f"System: {message['content']}\n\n"
    #     elif message["role"] == "user":
    #         prompt_text += f"User: {message['content']}\n\n"
    #     elif message["role"] == "assistant":
    #         prompt_text += f"Assistant: {message['content']}\n\n"

    # Tokenize and get token count before truncation

    print("Prompt:")
    print(prompt_text)
    tokens = TOKENIZER(prompt_text)
    num_tokens = len(tokens['input_ids'])
    
    # Tokenize and truncate to max length

    max_length = 4096
    inputs = TOKENIZER(
        prompt_text,
        return_tensors="pt",
        truncation=True, 
        max_length=max_length  # Increased from 1024 to handle longer prompts
    )
    
    if num_tokens > max_length:
        print(f"Warning: Input was truncated from {num_tokens} to {max_length} tokens")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Get initial prompt length to check only new tokens
    initial_length = len(inputs['input_ids'][0])
    # Disable gradient computation for generation
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=2048,  # Increased from 1000 for longer generations
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            stopping_criteria=[
                lambda input_ids, scores, **kwargs: bool(
                    TOKENIZER.decode(input_ids[0][initial_length:]).find("#end_of_example") != -1
                )
            ]
        )
    # Get the full generated text
    generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    # Find the last occurrence of "#to_be_implemented" and "#end_of_example"
    start_idx = generated_text.rfind("#to_be_implemented")
    end_idx = generated_text.rfind("#end_of_example")
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        # Extract the text between the markers and strip whitespace
        implementation = generated_text[start_idx + len("#to_be_implemented"):end_idx].strip()
        return implementation
    
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)