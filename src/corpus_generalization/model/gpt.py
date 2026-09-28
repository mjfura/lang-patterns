"""A decoder-only transformer in the GPT-2 layout: the usual small baseline for testing what a
corpus lets a language model learn, kept fixed across corpora so the corpus is the only variable.

Learned token and position embeddings; pre-LayerNorm blocks of causal self-attention and a 4x
GELU MLP; a final LayerNorm; an output head tied to the token embedding (GPT-2's choice, which
also keeps the parameter count of a small model from being dominated by two vocabulary-sized
matrices).
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head:
            raise ValueError(f"d_model {config.d_model} is not divisible by n_head {config.n_head}")
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        query, key, value = (
            part.view(batch, time, self.n_head, width // self.n_head).transpose(1, 2)
            for part in self.qkv(x).split(width, dim=2)
        )
        attended = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, time, width)
        projected: torch.Tensor = self.residual_dropout(self.projection(merged))
        return projected


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        output: torch.Tensor = x + self.mlp(self.mlp_norm(x))
        return output


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.context = config.context
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        # GPT-2's scaling of the residual projections by depth, so the residual stream's variance
        # does not grow with the number of blocks.
        for name, parameter in self.named_parameters():
            if name.endswith(("attention.projection.weight", "mlp.2.weight")):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token logits `(batch, time, vocab)` for token ids `(batch, time)`."""
        time = tokens.shape[1]
        if time > self.context:
            raise ValueError(f"sequence of {time} tokens exceeds the context of {self.context}")
        positions = torch.arange(time, device=tokens.device)
        x = self.embedding_dropout(
            self.token_embedding(tokens) + self.position_embedding(positions)
        )
        for block in self.blocks:
            x = block(x)
        logits: torch.Tensor = self.head(self.final_norm(x))
        return logits

    def parameter_count(self) -> int:
        """Distinct parameters: the tied head is counted once, with the embedding."""
        return sum(parameter.numel() for parameter in self.parameters())
