import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def contrastive_loss(embeddings1, embeddings2, device):
    # all embeddings should have dimension batch_size x embed_len

    norm_embeddings1 = embeddings1 / torch.norm(embeddings1, dim=1, keepdim=True)
    norm_embeddings2 = embeddings2 / torch.norm(embeddings2, dim=1, keepdim=True)


    pairwise_prod = norm_embeddings1 @ norm_embeddings2.T # bs x bs

    pos_mask = torch.eye(len(pairwise_prod)).to(device)
    neg_mask = torch.ones_like(pairwise_prod).to(device) - pos_mask

    assert pos_mask.shape == neg_mask.shape

    pos_weight = 0.7
    neg_weight = 0.3

    pos_average = (pairwise_prod * pos_mask).sum() / pos_mask.sum()
    neg_average = (pairwise_prod * neg_mask).sum() / neg_mask.sum()

    return pos_average * pos_weight - neg_average * neg_weight


def evaluate_batch(model, batch, device):
    source_code1, msk1, source_code2, msk2 = batch

    source_code1 = source_code1.to(device)
    msk1 = msk1.to(device)

    source_code2 = source_code2.to(device)
    msk2 = msk2.to(device)

    embeddings1 = model(source_code1, msk1)[0][:, 0]  # B x embedding_dim
    embeddings2 = model(source_code2, msk2)[0][:, 0]  # B x embedding_dim

    loss = contrastive_loss(embeddings1, embeddings2, device=device)

    return loss

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            loss = evaluate_batch(model, batch, device)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss


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

def train_model(model, optimizer, train_loader, val_loader, device, logger, n_epochs=10):
    model.train()  # Set the model to training mode

    print("MODEL params:", count_trainable_parameters(model))


    train_loss_history = []
    val_loss_history = []

    
    initial_params = [p.clone() for p in model.parameters()]

    for epoch in range(1, n_epochs + 1):
        
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}") as t:
            for batch in t:
                optimizer.zero_grad()

                loss = evaluate_batch(model, batch, device)
                loss.backward() 

                optimizer.step()

                epoch_loss += loss.item()
                train_loss_history.append(loss.item())

                logger.log({
                    "train_loss": loss.item(),
                }, step=len(train_loss_history))

                t.set_postfix(
                    train_loss=round(train_loss_history[-1], 2),
                    val_loss=round(val_loss_history[-1] if len(val_loss_history) > 0 else 0, 2)
                )

                if len(train_loss_history) % 40 == 0:
                    val_loss = validate(model, val_loader, device)
                    val_loss_history.append(val_loss)

                    logger.log({
                        "val_loss": val_loss,
                    }, step=len(train_loss_history))
                    # print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} - Average Training Loss: {avg_epoch_loss:.4f}")
    
    real_diff = 0
    for before, after in zip(initial_params, model.parameters()):
        real_diff += ((before - after) ** 2).sum()
    print("Real diff in models:", real_diff)

    return train_loss_history, val_loss_history