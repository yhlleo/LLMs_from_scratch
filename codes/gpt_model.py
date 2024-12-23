import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, dim_in, dim_out,
                context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert dim_out % num_heads == 0, \
            "dim_out must be divisible by num_heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def split_heads(self, x, batch_size, num_tokens):
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def forward(self, x):
        b, num_tokens, dim_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = self.split_heads(keys, b, num_tokens)
        queries = self.split_heads(queries, b, num_tokens)
        values = self.split_heads(values, b, num_tokens)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = (attn_weights @ values).transpose(1, 2)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.dim_out)

        context_vecs = self.out_proj(context_vecs)
        return context_vecs

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * x**3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiheadAttention(
            dim_in    = cfg["emb_dim"],
            dim_out   = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout   = cfg["drop_rate"],
            qkv_bias  = cfg["qkv_bias"]
        )

        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trans_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trans_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits