
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "num_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-media (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emd_dim": 1600, "n_layers": 48, "n_heads": 25}
}

def build_gpt2_configs(model_type, update_cfg):
    assert model_type in gpt2_model_configs, "Unknown model name!"

    new_config = GPT_CONFIG_124M.copy()
    new_config.update(gpt2_model_configs[model_type])

    new_config.update(update_cfg)
    return new_config
    