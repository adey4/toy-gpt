import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

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
class BigramLanguageModel(nn.Module):
    # predictions for next token are made based only on current token
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads logits for next token from lookup table.
        # each token has a unique row of size vocab_size to read logits from.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # idx and targets are both int tensors of shape (Batch, Time).
        # for each token in each context window 
        # (Time -> num. of tokens in context window) 
        # in each chunk (Batch -> number of chunks in batch),
        # pull out the corresponding logits (Channel -> vocab_size/feature size)
        # from the lookup table
        logits = self.token_embedding_table(idx) # (Batch, Time, Channel)
        
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
            # get predictions
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# %%
# create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %%
# train the Bigram model
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
