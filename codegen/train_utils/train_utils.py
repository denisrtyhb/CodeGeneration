import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.cuda.amp
def contrastive_loss(embeddings1, embeddings2, embeddings3, device, val=False):
    assert embeddings1.shape == embeddings2.shape
    assert embeddings1.shape == embeddings3.shape

    positive_sim = F.cosine_similarity(embeddings1, embeddings2)
    negative_sim = F.cosine_similarity(embeddings1, embeddings3)
    
    delta = negative_sim - positive_sim + 0.2
    # negative_sim - positive_sim + 0.2 < 0
    # negative_sim + 0.2 < positive_sim
    delta = torch.clamp(delta, min=0)

    if val:
        print("Easy, medium, hard", (delta == 0).sum().item(), (delta > 0).sum().item(), (delta > 0.2).sum().item(), "out of", len(delta))

    return delta.mean()

def contrastive_loss_and_metrics(embeddings1, embeddings2, embeddings3):
    assert embeddings1.shape == embeddings2.shape
    assert embeddings1.shape == embeddings3.shape

    positive_sim = F.cosine_similarity(embeddings1, embeddings2)
    negative_sim = F.cosine_similarity(embeddings1, embeddings3)
    
    delta = negative_sim - positive_sim + 0.2
    delta = torch.clamp(delta, min=0)

    metrics = {
        "val_loss": delta.sum().item(),
        "easy_triplets": (delta == 0).sum().item(),
        "medium_triplets": (delta > 0).sum().item(),
        "hard_triplets": (delta > 0.2).sum().item(),
    }
    return metrics

def evaluate_triplet_batch(model, batch, device, val=False):
    source_code1, msk1, source_code2, msk2, source_code3, msk3 = batch

    # Convert inputs to bfloat16
    source_code1 = source_code1.to(device)
    msk1 = msk1.to(device)

    source_code2 = source_code2.to(device)
    msk2 = msk2.to(device)

    source_code3 = source_code3.to(device)
    msk3 = msk3.to(device)  
    
    batch_size = source_code1.size(0)
    seq_len = source_code1.size(1)
    hidden_size = model.model.config.hidden_size
    
    input_memory = 2 * (batch_size * seq_len * 2)
    
    n_layers = model.model.config.num_hidden_layers
    hidden_memory = 2 * batch_size * seq_len * hidden_size * 4 * (n_layers + 1)
    
    n_heads = model.model.config.num_attention_heads
    attention_memory = batch_size * n_heads * seq_len * seq_len * 4 * n_layers
    
    total_memory = (input_memory + hidden_memory + attention_memory) / (1024**3)
    
    if total_memory > torch.cuda.get_device_properties(device).total_memory / (1024**3):
        print(f"Warning: Estimated memory usage ({total_memory:.2f}GB) exceeds available GPU memory")
    elif total_memory > torch.cuda.get_device_properties(device).total_memory / (1024**3) * 0.8:
        print(f"Warning: Estimated memory usage ({total_memory:.2f}GB) exceeds 80% of available GPU memory")

    embeddings1 = model(source_code1, msk1)  # B x embedding_dim
    embeddings2 = model(source_code2, msk2)  # B x embedding_dim
    embeddings3 = model(source_code3, msk3)  # B x embedding_dim
    loss = contrastive_loss(embeddings1, embeddings2, embeddings3, device=device, val=val)

    return loss
def evaluate_and_calc_metrics(model, batch, device):
    source_code1, msk1, source_code2, msk2, source_code3, msk3 = batch

    embeddings1 = model(source_code1.to(device), msk1.to(device))  # B x embedding_dim
    embeddings2 = model(source_code2.to(device), msk2.to(device))  # B x embedding_dim
    embeddings3 = model(source_code3.to(device), msk3.to(device))  # B x embedding_dim

    metrics = contrastive_loss_and_metrics(embeddings1, embeddings2, embeddings3)
    return metrics

def validate(model, val_loader, device):
    model.eval()
    total_metrics = {}
    total_length = 0
    with torch.no_grad():
        for batch in val_loader:
            metrics = evaluate_and_calc_metrics(model, batch, device)
            total_length += len(batch[0])
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            assert total_metrics['easy_triplets'] + total_metrics['medium_triplets'] == total_length
    for key, value in total_metrics.items():
        total_metrics[key] = value / total_length
    print("Validation info:", total_metrics, total_length)
    model.train()
    return total_metrics


def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters (parameters with requires_grad=True) in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_trainable_params

def sum_gradient_norms(model):
    """
    Calculates the sum of the L2 norms of the gradients of all parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: The sum of the gradient norms. Returns 0 if no gradients are present.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item()  # Accumulate the norm

    return total_norm

def train(model, optimizer, train_loader, val_loader, device, logger, n_epochs=10, accumulation_steps=1):
    model.train()  # Set the model to training mode

    print("MODEL params:", count_trainable_parameters(model))


    train_loss_history = []
    val_loss_history = []

    # scaler = torch.cuda.amp.GradScaler()

    i = 0
    for epoch in range(1, n_epochs + 1):
        
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}") as t:
            for batch in t:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = evaluate_triplet_batch(model, batch, device)

                loss.backward()
                i += 1
                if i % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                train_loss_history.append(loss.item())

                logger.log({
                    "train_loss": loss.item(),
                }, step=len(train_loss_history))

                t.set_postfix(
                    train_loss=round(train_loss_history[-1], 2),
                    val_loss=round(val_loss_history[-1] if len(val_loss_history) > 0 else 0, 2)
                )

                if len(train_loss_history) % 50 == 0:
                    metrics = validate(model, val_loader, device)
                    val_loss_history.append(metrics["val_loss"])
                    print("val loss", val_loss_history[-1])
                    logger.log(metrics, step=len(train_loss_history))
                    # print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} - Average Training Loss: {avg_epoch_loss:.4f}")

    return train_loss_history, val_loss_history