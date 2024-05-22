import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniRogue(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(n_embd, n_head, n_embd*4, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        b, t = x.size()
        token_embeddings = self.tok_emb(x)  # (b, t, n_embd)
        position_embeddings = self.pos_emb[:, :t, :]  # (1, t, n_embd)
        x = token_embeddings + position_embeddings  # (b, t, n_embd)
        x = self.blocks(x)  # (b, t, n_embd)
        x = self.ln_f(x)  # (b, t, n_embd)
        logits = self.head(x)  # (b, t, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

def get_model(vocab_size, block_size, n_embd=128, n_head=2, n_layer=2, dropout=0.1):
    return MiniRogue(vocab_size, block_size, n_embd, n_head, n_layer, dropout)