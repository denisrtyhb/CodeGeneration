"""
Main module for the code generation framework.
"""
import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from codegen.train_utils.models import load_model_and_tokenizer
import argparse
from tqdm import tqdm

SOURCE_CODE_COLUMN = "raw_source_code"
EMBEDDING_COLUMN = "embeddings"

from prompt_creation import create_prompt_with_context, create_prompt_without_context
from openai import OpenAI

# get api_key from environment variable
api_key = os.getenv("OPENAI_API_KEY").strip()
assert api_key is not None, "OPENAI_API_KEY is not set"
# api_key = "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"
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key
)

ALWAYS_CONTINUE = False
def generate_code_with_smart_model(prompt):
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-qwen-14b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response_text = response.choices[0].message.content
    if "```python" in response_text:
        response_text = response_text.split("```python")[1].split("```")[0]
    global ALWAYS_CONTINUE
    if ALWAYS_CONTINUE:
        return response_text
    print("Code:\n",response_text)
    command = input("continue?")
    if command == "always":
        ALWAYS_CONTINUE = True
        return response_text
    if command == "y":
        return response_text
    exit()

def find_closest_embeddings(embedding, embeddings, k=5):

    # find 5 closest embeddings to the given embedding
    # cast column to tensor
    distances = torch.cdist(embedding.reshape(1, -1), embeddings)   

    # get the indices of the 5 smallest distances, excluding the 0th index
    closest_embeddings = torch.argsort(distances, dim=1)[:, 1:k+1]
    return closest_embeddings.flatten().tolist()

MODEL, TOKENIZER = None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model(model_name, pretrain_path=None):
    global MODEL, TOKENIZER
    #model from train_utils.models.py
    MODEL, TOKENIZER = load_model_and_tokenizer(model_name)
    if pretrain_path:
        MODEL.load_model(pretrain_path)
    MODEL.to(device)

@torch.no_grad()
def embed_code(code):
    global MODEL, TOKENIZER
    assert MODEL is not None, "Model not loaded"
    inputs = TOKENIZER(code, return_tensors="pt").to(device)
    outputs = MODEL(**inputs)[0].cpu().to(float)
    
    # print("Output type: ", type(outputs))
    return outputs

def evaluate(dataset_path: str, output_path: str, model_path: str, model_name: str):
    """
    
    Args:
        dataset_path (str): Path to the dataset
        model_path (str): Path to the model checkpoint
        output_path (str): Path to save the evaluation results
        model_name (str): Name of the model to use
    """
    # Load dataset using train_utils.data_utils.load_dataset 
    dataframe_path = os.path.join(dataset_path, "function_base.csv")
    dataset = pd.read_csv(dataframe_path)
    if SOURCE_CODE_COLUMN not in dataset.columns:
        raise ValueError(f"Column {SOURCE_CODE_COLUMN} not found in dataset")
    
    # Calculate embeddings for all rows
    if model_name is not None:
        embeddings = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Calculating embeddings"):
            embeddings.append(embed_code(row[SOURCE_CODE_COLUMN]))
        embeddings = torch.stack(embeddings)
    # Generate code for all rows

    results = []

    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating code"):

        if model_name is None:
            no_context_generation = generate_code_with_smart_model(
                create_prompt_without_context(row["function signature"], row["comment"]))
            results.append({
                "function signature": row["function signature"],
                "source code": row[SOURCE_CODE_COLUMN],
                "no_context_generation": no_context_generation,
            })
        else:
            embedding = row[EMBEDDING_COLUMN]
            # find 5 closest embeddings to the given embedding
            closest_embeddings = find_closest_embeddings(embeddings[i], embeddings)
            closest_rows = dataset.iloc[closest_embeddings]

            # print(closest_rows.columns, closest_rows)
            
            context_generation = generate_code_with_smart_model(
                create_prompt_with_context(
                    row["function signature"],
                    row["comment"],
                    closest_rows.iterrows()))

            results.append({
                "function signature": row["function signature"],
                "source code": row[SOURCE_CODE_COLUMN],
                "context_generation": context_generation,
            })
    results = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    results.to_csv(os.path.join(output_path, "results.csv"), index=False)
    

def evaluate_all_datasets(dataset_path, model_path, model_name, output_path):
    if model_name is not None:
        _load_model(model_name, model_path)
    for dataset_name in os.listdir(dataset_path):
        evaluate(
            dataset_path=os.path.join(dataset_path, dataset_name),
            model_path=model_path,
            model_name=model_name,
            output_path=os.path.join(output_path, dataset_name)
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Code Generation Framework')
    
    parser.add_argument('--dataset_path', type=str, default="framework_dataset/",
                      help='Path to the dataset directory')
    
    parser.add_argument('--model_path', type=str,
                      help='Path to the model checkpoint')
    
    parser.add_argument('--model_name', type=str, default="ibm-granite/granite-3.1-1b-a400m-base",
                      help='Name of the model to use')
    
    parser.add_argument('--output_path', type=str, default="results/result.json",
                      help='Path to save the evaluation results')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_all_datasets(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        model_name=args.model_name,
        output_path=args.output_path
    )

