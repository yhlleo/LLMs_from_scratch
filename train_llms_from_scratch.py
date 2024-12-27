import torch
import tiktoken

from codes.configs import GPT_CONFIG_124M
from codes.gpt_model import GPTModel
from codes.data import build_dataloader
from codes.solver import train_model_simple
from codes.plots import plot_losses

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
    torch.manual_seed(123)
    
    device = torch.device("cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    train_loader, val_loader = build_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=4e-4, weight_decay=0.1)

    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10
    train_losses, val_losses, token_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, token_seen, train_losses, val_losses)
