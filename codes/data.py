import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken

from configs import GPT_CONFIG_124M

class GPTDatasetV2(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
	text, 
	batch_size=4, 
	max_length=256, 
	stride=128, 
	shuffle=True, 
	drop_last=True, 
	num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV2(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def build_dataloader():
    with open("../the-verdict.txt", "r", encoding="utf-8") as fin:
        raw_data = fin.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(raw_data))
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=True
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=False,
        shuffle=False
    )
    return train_loader, val_loader