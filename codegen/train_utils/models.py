import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM

class CodeEmbedder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load the base model without classification head
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.train()
        self.model.to(torch.bfloat16)
        # Freeze the model parameters if needed
        # for param in self.model.parameters():
        #     param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        # Get the model outputs


        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        # print(embeddings.shape)
        # while True:
        #     exec(input())
        # embeddings = outputs[0][:, 0, :]

        # print("Number of nans:", torch.isnan(embeddings).sum())
        
        return embeddings
    def save_model(self, save_path):
        """
        Save the model state dictionary to a file.
        
        Args:
            save_path (str): Path where the model should be saved
        """
        torch.save(self.state_dict(), save_path)
        
    def load_model(self, load_path):
        """
        Load the model state dictionary from a file.
        
        Args:
            load_path (str): Path to the saved model file
        """
        self.load_state_dict(torch.load(load_path))
        self.eval()

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, 'pad_token'):
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with bfloat16 precision
    model = CodeEmbedder(model_name)
    print(model)
    return model, tokenizer


def test_model_loading(model_name):
    print(f"Testing model loading with {model_name}")
    
    # Test model and tokenizer loading
    model, tokenizer = load_model_and_tokenizer(model_name)
    assert isinstance(model, CodeEmbedder), "Model should be an instance of CodeEmbedder"
    
    # Test tokenizer has required attributes
    assert hasattr(tokenizer, 'pad_token'), "Tokenizer should have pad_token"
    
    return model, tokenizer

def test_single_input(model, tokenizer):
    # Create sample input
    sample_code = "def test(): return True"
    inputs = tokenizer(sample_code, return_tensors="pt", padding=True, truncation=True)
    
    # Test forward pass
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Validate output shape and type
    assert isinstance(outputs, torch.Tensor), "Output should be a torch.Tensor"
    assert len(outputs.shape) == 2, "Output should be 2-dimensional (batch_size x hidden_size)"
    assert outputs.shape[0] == 1, "Batch size should be 1 for single input"
    
    return outputs

def test_batch_input(model, tokenizer, outputs):
    # Test with batched inputs
    batch_codes = [
        "def test1(): return True",
        "def test2(): return False",
        "class MyClass: pass",
        "print('Hello World')"
    ]
    batch_inputs = tokenizer(batch_codes, return_tensors="pt", padding=True, truncation=True)
    
    # Test batched forward pass
    batch_outputs = model(batch_inputs['input_ids'], batch_inputs['attention_mask'])
    
    # Validate batched output
    assert isinstance(batch_outputs, torch.Tensor), "Batched output should be a torch.Tensor"
    assert len(batch_outputs.shape) == 2, "Batched output should be 2-dimensional (batch_size x hidden_size)"
    assert batch_outputs.shape[0] == len(batch_codes), f"Batch size should match number of inputs ({len(batch_codes)})"
    assert batch_outputs.shape[1] == outputs.shape[1], "Hidden size should be consistent between batched and single outputs"
    
    print(f"Batched output shape: {batch_outputs.shape}")
    
    return batch_outputs

if __name__ == "__main__":
    model_name = "ibm-granite/granite-8B-code-base-128k"
    
    model, tokenizer = test_model_loading(model_name)
    outputs = test_single_input(model, tokenizer)
    batch_outputs = test_batch_input(model, tokenizer, outputs)
    
    print("All tests passed successfully!")
    print(f"Output embedding shape: {outputs.shape}")

