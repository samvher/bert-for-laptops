import torch
import torch.nn as nn

from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    Bi-directional transformer self-attention.
    Uses relative position embeddings, shared across tokens and attention heads, but unique for each layer.
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        embed_size = config["embed_size"]
        n_head = config["n_head"]
        assert embed_size % n_head == 0
        
        # This is clipping distance (k) in Shaw et al
        pos_emb_radius = config["pos_emb_radius"]
        pos_emb_units = config["embed_size"] // config["n_head"]
        
        # Position embedding vectors for use on keys
        # This is w^K in Shaw et al
        self.pos_emb_k = nn.Parameter(torch.zeros(2 * pos_emb_radius, pos_emb_units))
        torch.nn.init.normal_(self.pos_emb_k, mean=0.0, std=0.02)
        
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_size, embed_size, bias = False)
        self.query = nn.Linear(embed_size, embed_size, bias = False)
        self.value = nn.Linear(embed_size, embed_size, bias = False)
        
        # output projection
        self.proj = nn.Linear(embed_size, embed_size, bias = False)

    def forward(self, x):
        batch_size, context_size, embed_size = x.size()
        assert embed_size == self.config["embed_size"]
        
        n_head = self.config["n_head"]
        head_size = embed_size // n_head
        
        pos_emb_size, head_size = self.pos_emb_k.size()
        pos_emb_radius = self.config["pos_emb_radius"]
        assert pos_emb_size == 2 * pos_emb_radius

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(batch_size, context_size, n_head, head_size).transpose(1, 2) # (batch_size, n_head, context_size, head_size)
        q = self.query(x).view(batch_size, context_size, n_head, head_size).transpose(1, 2) # (batch_size, n_head, context_size, head_size)
        v = self.value(x).view(batch_size, context_size, n_head, head_size).transpose(1, 2) # (batch_size, n_head, context_size, head_size)
        
        # Below section implements x_i W^Q (a_{ij}^K)^T from Shaw et al
        # position attention: (batch_size, n_head, context_size, head_size) x (1, 1, pos_emb_size, head_size) -> (batch_size, n_head, context_size, pos_emb_size)
        att_rel_pos = q @ self.pos_emb_k.view(1, 1, pos_emb_size, head_size).transpose(-2, -1)
        att_idxs = (torch.clamp(torch.arange(context_size)[None, :] - torch.arange(context_size)[:, None], -pos_emb_radius, pos_emb_radius-1) % pos_emb_size).to("cuda")
        att_pos = torch.gather(att_rel_pos, 3, att_idxs.expand((batch_size, n_head, context_size, context_size)))
        assert att_pos.shape == (batch_size, n_head, context_size, context_size)
        
        # value attention: (batch_size, n_head, context_size, head_size) x (batch_size, n_head, context_size, head_size) -> (batch_size, n_head, context_size, context_size)
        att_val = q @ k.transpose(-2, -1)
        
        # combined attention
        att_scale = 1.0 / math.sqrt(k.size(-1))
        att = F.softmax((att_val + att_pos) * att_scale, dim=-1) # Equation (5) from Shaw et al
        
        y = att @ v # (batch_size, n_head, context_size, context_size) x (batch_size, n_head, context_size, head_size) -> (batch_size, n_head, context_size, head_size)
        y = y.transpose(1, 2).contiguous().view(batch_size, context_size, embed_size) # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

    
class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, config):
        super().__init__()
        
        embed_size = config["embed_size"]
        
        self.norm1 = nn.LayerNorm(embed_size, eps = 1e-6)
        self.attn = SelfAttention(config)
        
        self.norm2 = nn.LayerNorm(embed_size, eps = 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size, bias = False),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size, bias = False),
        )

    def forward(self, x):
        # This is Pre-LayerNorm
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        # Post-LayerNorm would look more like
        # x = self.norm1(x + self.attn)
        # x = self.norm2(x + self.mlp)
        
        return x

    
class BERT(nn.Module):
    """Headless BERT."""

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        vocab_size = config["vocab_size"]
        embed_size = config["embed_size"]
        n_layer = config["n_layer"]

        # token embedding
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.norm_emb = nn.LayerNorm(embed_size, eps = 1e-6)
        
        # transformer
        self.transformer = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        
        # final layernorm
        self.norm_final = nn.LayerNorm(embed_size, eps = 1e-6)

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        batch_size, context_size = x.size()
    
        x = self.tok_emb(x)
        x = self.norm_emb(x)
        x = self.transformer(x)
        x = self.norm_final(x)
        
        return x

    
class MLMHead(nn.Module):
    """
    BERT head for masked language modeling.
    Note that this does *not* implement sparse prediction as mentioned in the Cramming paper. Predictions are calculated for all tokens.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        vocab_size = config["vocab_size"]
        embed_size = config["embed_size"]
        
        self.tok_unemb = nn.Linear(embed_size, vocab_size, bias = False)
    
    def forward(self, x, y):
        logits = self.tok_unemb(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index = 0)
        return logits, loss
    

class CLSHead(nn.Module):
    """
    BERT head for classification.
    A prediction is only calculated for the first ([CLS]) token.
    """
    
    def __init__(self, config, n_classes):
        super().__init__()
        
        self.config = config
        embed_size = config["embed_size"]
        
        self.classifier = nn.Linear(embed_size, n_classes)
    
    def forward(self, x, y = None):
        logits = self.classifier(x[:, 0, :])
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss


class RegHead(nn.Module):
    """
    BERT head for regression.
    A prediction is only calculated for the first ([CLS]) token.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        embed_size = config["embed_size"]
        
        self.regressor = nn.Linear(embed_size, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, y = None):
        y_hat = self.regressor(x[:, 0, :])
        loss = None
        if y is not None:
            loss = self.loss_fn(y_hat.view(-1), y.view(-1))
        return y_hat, loss