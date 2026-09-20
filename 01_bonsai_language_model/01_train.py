"""Train a minimal decoder-only Transformer language model in plain PyTorch.

No nn.Transformer, nn.MultiheadAttention, or external model library is used.
The deliberately tiny model has one causal self-attention head and one block.

Run to train and save a checkpoint.
"""

import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = pathlib.Path(__file__).parent / "checkpoint.pt"

# -----------------------------------------------------------------------------
# 1. Data: characters are our tokens
# -----------------------------------------------------------------------------

torch.manual_seed(0)

TEXT = """
the robot picked up the red cup.
the robot put down the blue cup.
the small robot moved to the table.
the red cup was on the table.
the blue cup was near the robot.
""" * 100

characters = sorted(set(TEXT))
vocab_size = len(characters)
char_to_id = {character: i for i, character in enumerate(characters)}
id_to_char = {i: character for character, i in char_to_id.items()}


def encode(text: str) -> list[int]:
    return [char_to_id[character] for character in text]


def decode(token_ids: list[int]) -> str:
    return "".join(id_to_char[token_id] for token_id in token_ids)


data = torch.tensor(encode(TEXT), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
validation_data = data[n_train:]


batch_size = 16
context_length = 32
embedding_dimension = 48


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inputs and their next-character targets, both shaped [B, T]."""
    source = train_data if split == "train" else validation_data
    starts = torch.randint(len(source) - context_length - 1, (batch_size,))

    x = torch.stack([source[i : i + context_length] for i in starts])
    y = torch.stack([source[i + 1 : i + context_length + 1] for i in starts])
    return x, y


# -----------------------------------------------------------------------------
# 2. One causal self-attention head
# -----------------------------------------------------------------------------

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, dimension: int, max_context_length: int):
        super().__init__()

        # These layers contain the learned W_Q, W_K, and W_V matrices.
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)

        # Lower-triangular mask: position i may only read positions j <= i.
        causal_mask = torch.tril(
            torch.ones(max_context_length, max_context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]
        _, time, channels = x.shape

        q = self.query(x)                    # [B, T, C]
        k = self.key(x)                      # [B, T, C]
        v = self.value(x)                    # [B, T, C]

        # Every query position is compared with every key position.
        scores = q @ k.transpose(-2, -1)     # [B, T, T]
        scores = scores / math.sqrt(channels)

        # Future positions receive -infinity, hence zero probability after softmax.
        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # [B, T, T]

        # Each position receives a weighted sum of value vectors.
        output = attention_weights @ v       # [B, T, C]
        return output


# -----------------------------------------------------------------------------
# 3. One Transformer block: attention, MLP, and residual connections
# -----------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dimension: int, max_context_length: int):
        super().__init__()
        self.norm_before_attention = nn.LayerNorm(dimension)
        self.attention = CausalSelfAttentionHead(dimension, max_context_length)

        self.norm_before_mlp = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.ReLU(),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path: preserve x and add the transformation to it.
        x = x + self.attention(self.norm_before_attention(x))
        x = x + self.mlp(self.norm_before_mlp(x))
        return x


# -----------------------------------------------------------------------------
# 4. Complete decoder-only language model
# -----------------------------------------------------------------------------

class MinimalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Maps token IDs and positions to vectors of the same dimension.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)

        self.block = TransformerBlock(embedding_dimension, context_length)
        self.final_norm = nn.LayerNorm(embedding_dimension)

        # Converts each final token vector into one score per vocabulary item.
        self.vocabulary_projection = nn.Linear(embedding_dimension, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        _, time = token_ids.shape

        token_vectors = self.token_embedding(token_ids)       # [B, T, C]
        positions = torch.arange(time, device=token_ids.device)
        position_vectors = self.position_embedding(positions) # [T, C]

        x = token_vectors + position_vectors                  # [B, T, C]
        x = self.block(x)                                     # [B, T, C]
        x = self.final_norm(x)                                # [B, T, C]
        logits = self.vocabulary_projection(x)                # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(self, initial_tokens: torch.Tensor, number_of_tokens: int) -> torch.Tensor:
        """Autoregressively append one sampled token at a time."""
        self.eval()
        tokens = initial_tokens

        for _ in range(number_of_tokens):
            # The model only has positional embeddings for context_length places.
            visible_tokens = tokens[:, -context_length:]

            logits = self(visible_tokens)             # [B, T, V]
            next_token_logits = logits[:, -1, :]       # [B, V]
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # [B, 1]
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens


# -----------------------------------------------------------------------------
# 5. Training: predict every next token in parallel
# -----------------------------------------------------------------------------

def train() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MinimalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"device: {device}")
    print(f"vocabulary: {vocab_size} characters")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    model.train()
    for step in range(1001):
        inputs, targets = get_batch("train")
        inputs, targets = inputs.to(device), targets.to(device) # Input is a text window of length T, and target a window of same size advanced by one.

        # Because of causal masking, this single (inputs, targets) window doesn't give just one
        # training example — it gives T, one per prefix length. This is what makes autoregressive
        # training efficient: instead of running the model once per prefix (T forward passes), one
        # forward pass over the full window produces all T predictions at once, each mathematically
        # identical to what you'd get by feeding that prefix alone.
        logits = model(inputs)  # [B, T, V]; logits[:, t, :] are the vocabulary scores predicted from context inputs[:, 0:t+1]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
