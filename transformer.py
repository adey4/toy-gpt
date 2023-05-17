import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)
print(f"Running on {device}")

# %%
# download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# %%
# read in the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# %%
# get all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# %%
# map characters to integers (tokenizer)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # input: string, output: list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # input: list of ints,
                                                 # output: string


# %%
# encode the entire dataset and store in a PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# %%
# train-validation split
n = int(0.9*(len(data))) # first 90% of data will be training data
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %%
class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention score (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

    
# %%
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # project back into residual pathway
        return out
    
    
# %%
class FeedForward(nn.Module):
    """ simple linear layer + nonlinearity (ReLu) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # project back to residual pathway
            nn.Dropout(dropout)  # dropout allows us to train a "ensemble"-like
                                 # collection of subnets that are aggregated 
                                 # for inference
        )
        
    def forward(self, x):
        return self.net(x)
    
    
# %%
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of sa heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # layer norm acts as per-token tranformation
                                         # that normalizes features initially
                                         # with guassian 0 mean, unit var distr.
                                         # normalize over all channels per token
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # "x +" introduces residual pathway that helps with
                                      # optimization of gradients (direct pathway from)
                                      # inputs to outputs
        x = x + self.ffwd(self.ln2(x))
        return x
        

# %%
class GPT(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token has a unique row of size vocab_size to read logits from.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim self-attention.
                                                         # multiple heads are useful
                                                         # because tokens need to communicate
                                                         # lots of info.
                                                         # e.g. what vowels are interesting?
                                                         # what consonants are interesting?
                                                         # what vowels at a certain position
                                                         # are interesting?
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both int tensors of shape (Batch, Time).
        # for each token in each context window 
        # (Time -> num. of tokens in context window) 
        # in each chunk (Batch -> number of chunks in batch),
        # pull out the corresponding logits (Channel -> vocab_size/feature size)
        # from the lookup table
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channel=n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        #x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
        #x = self.ffwd(x) # computation on per-node (per-token) level (B, T, C)
                         # allows tokens to "think" about info gathered from
                         # communication thru self-attention
        logits = self.lm_head(x) #(B, T, vocab_size)
        
        # measure quality of logits based on target (how well are we predicting
        # next character)
        if targets is None:
            loss = None
        else:
            # reshape logits for pytorch: (Batch, Channel, Time)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        # logits represent probability that a character is the correct next-word
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context window.
        # we are generating B next-word predictions and appending each
        # prediction to the appropriate context window in the batch B
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)
# print number of parameters in model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# %%
# create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
# train the model
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},"
              f" val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# %%
