import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def contrastive_loss(embeddings1, embeddings2, labels):
    cos_sim = F.cosine_similarity(embeddings1, embeddings2)
    loss = torch.mean((1 - labels) * cos_sim +
                      (labels) * torch.clamp(1 - cos_sim, min=0.0))
    return loss


def evaluate_batch(model, batch, device):
    source_code1, msk1, source_code2, msk2 = batch

    source_code1 = source_code1.to(device)
    msk1 = msk1.to(device)

    source_code2 = source_code2.to(device)
    msk2 = msk2.to(device)

    embeddings1 = model(source_code1, msk1)[0][:, 0]  # B x embedding_dim
    embeddings2 = model(source_code2, msk2)[0][:, 0]  # B x embedding_dim

    labels = torch.ones(len(msk1)).to(device) # All pairs are connected - positive examples for the dataset

    loss = contrastive_loss(embeddings1, embeddings2, labels)

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

def train_model(model, optimizer, train_loader, val_loader, device, logger, n_epochs=10):
    model.train()  # Set the model to training mode

    epoch_loss = 0.0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, n_epochs + 1):
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
    return train_loss_history, val_loss_history