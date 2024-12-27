import torch
import tiktoken

from .gpt_model import GPTModel
from .data import build_dataloader
from .losses import calc_loss_batch, evaluate_model
from .utils import generate_and_print_sample
from .configs import GPT_CONFIG_124M

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def train_model_simple(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    device, 
    num_epochs, 
    eval_freq, 
    eval_iter,
    start_context,
    tokenizer
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(token_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} "
                      f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


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

    def plot_losses(epochs_seen, token_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax1.twiny()
        ax2.plot(token_seen, train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")
        fig.tight_layout()
        plt.show()

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, token_seen, train_losses, val_losses)
