# Making a Vision Transformer

The last blog I made focused making a text-based decoder only transformer. 
Today, I want to take advantage of the fact that the concepts in a vision transformer
are very similar to that of a text transformer and try to implement a vision transformer based on the paper, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al.

## Summary of a text transformer (decoder-only) architecture

As talked about in my previous post, a text transformer has the following general parts, roughly starting from smallest to largest:

### 1. Token embedding table, applied to input immediately

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```
### 2. Position embedding table, applied to input immediately (to provide position info about tokens)

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

### 3. Single attention head, 1 head/multi-head attention

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(vocab_size, head_size, bias = False)
        self.query = nn.Linear(vocab_size, head_size, bias = False)
        self.value = nn.Linear(vocab_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        #add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, targets):
        B, T, C = x.shape
        k = self.key(idx)
        q = self.query(idx)
        v = self.value(idx)
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)

        #apply dropout to randomly prevent some nodes from communicating
        wei = self.dropout(wei)

        out = wei @ v
        return out
```

### 4. Multi-headed attention, 1 part/block

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

        #add dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, targets):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        #add dropout
        out = self.dropout(self.proj(out))
        return out

```

### 5. Feed forward, 1 part/block (applied after multi-headed attention to allow for "thinking" per node to occur)
```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
            nn.Dropout(dropout) #add dropout
        )
    def forward(self, x):
        return self.net(x)
```

### 6. Block, many/ 1 model


```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        #initialize layernorms
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+ self.sa(self.ln1(x)) #apply layernorms before feeding x into the attention heads

        x = x+ self.ffwd(self.ln2(x))
        return x
```

### 7. Overall Model 

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #make number of blocks and number of heads variable
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size) 

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) #pass x through layernorm
        logits = self.lm_head(x)
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

This image is again very good for visualizing this nested structure (minus the fact that in our decoder-only model there is no cross-attention between the encoder and the decoder):

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/4e44c7e2-9113-48fb-b499-3d40996d3786">

## Vision Transformer: Motivations

So, I wanted to first summarize some of the motivations of the authors based on the introduction of the vision transformer paper.

Basically, the authors talk about how in 2020, transformers were the architecture of choice in NLP-land because of efficiency and scalability but in computer vision CNNs were still dominant.
Before this paper, multiple works have tried combining CNNs with attention as inspired by
attention's success in NLP-land. However, the authors claim that these past works don't really
incorporate the scaling that is essential to text-based transformers' success, and so they
try to do just that, in "applying a standard Transformer directly to images with fewest possible modifications."

What is pretty cool is that they find that while ResNet performs better on mid-sized datasets, the Vision Transformer (ViT)
approaches or beats state-of-the-art results on larger datsets (14M-300M). They state this as "large scale training trumps inductive bias".

## General Model architecture

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/332671db-3fbb-419d-88a3-d1ff8aaaedb1">

This figure is taken directly from the paper, and immediately we can notice that like the authors intended,
the architecture of the encoder is identical to that of an encoder of a text-based transformer. Like we implemented before, in each block out of L total in the encoder, the input is fed into a layer norm then the multi-head attention, with a skip connection 
after the multi-head attention is applied. Then, the residual stream is fed again into a layer norm, then a feed forward/MLP layer, then combined back
into the residual stream. 

There are however slight changes in what is fed into the transformer encoder and how the output is generated.

In particular, because the input data is now image-based, the authors needed a way to efficiently embed images. This seems to be the primary innovation of this paper, so we'll focus the most time on this.

After the embedded images are passed through the transformer encoder, there is also an extra MLP head added to project the encodings to a specific class, which we will also discuss in a bit.

## Pre-transformer image encoding

This is the crux of the paper, which is how to convert image data into sequential data which is used in a transformer. There's a couple of innovations
the authors had here, which we will go through.

### Patches

The intuition behind patches is that just like converting text into tokens chops up text into units which hold some semantic meaning, 
the vision transformer in this paper chops up images into smaller "patches" or areas which may communicate with each other to aggregate 
local information to information about the image at large.

In terms of implementation details, 2D images are reshaped from (H,W,C) to (N, P^2 * C), or each of N patches is "unrolled" to a vector. 

2D images have generally height (H), width (W), and channels (C), which is generally 3 due to RGB. P represents the resolution of each patch (P x P part of the image), and N is the resulting number of patches. 
N can be calculated with the equation H*W/P^2. 
This is relatively straightforward to see why: if the image has HxWxC values originally, the reshaped image should still have that many values.
Therefore, N x P^2 x C = HxWxC so N must equal HxW/P^2. According to the original paper, N "serves as the effective input sequence length for the Transformer", which makes sense since we are feeding in N "visual tokens" to the transformer.

Here's some Python code to do this:

```python
import torch

H, W, C = 256, 256, 3
img = torch.rand(H, W, C)
P = 16
N = H*W//P**2
patches = img.reshape(N, P**2*C)
```

With a batch dimension, the code looks near identical since the batch dimension is unaffected by the reshaping:

```python
import torch

B, H, W, C = 2, 256, 256, 3
img = torch.rand(H, W, C)
P = 16
N = H*W//P**2
patches = img.reshape(B, N, P**2*C)
```

We can check that the number of values are the same using `torch.prod`:

```python
torch.prod(torch.tensor(img.shape)), torch.prod(torch.tensor(patches.shape))
```
returns (tensor(196608), tensor(196608))  (or (tensor(393216), tensor(393216)) with the batch dimension), which means that there is no loss of information.

### Position Embeddings
The next step is to add position embeddings to these patches.
The paper uses "standard learnable 1D position embeddings", which the authors find perform similarly to 2D position embeddings.
Essentially, what this means is that once an image is chopped up into patches, the patches are labeled only by their position in the sequence rather than their x,y coordinates relative to the original image.
These labels are fed into a position encoding table, just like we implemented for a text transformer.

This table looks something like this:

```python
#each possible position in sequence of N patches retrieves a D dimensional vector
self.position_embedding_table = nn.Embedding(N, D)

#or the more common implementation, which is
self.pos_emb = nn.Parameter(torch.zeros(1, N, D))
#which can then be initalized to weights in a normal distribution as
nn.init.normal_(self.pos_emb, std = 1e-6)
```

And in the forward function of our model, we can call the following to encode our patch sequence:

```python
pos_emb = self.position_embedding_table(torch.arange(N, device = device))

#or just self.pos_emb if using the nn.Parameter implementation.
```


### Projecting to fixed dimension size

So, after we have both the patch and position embeddings, we want to concatenate them, but their shapes are currently incompatible. Thus, we must project the patches to the same dimension D that the position embeddings are. This is done with a "trainable linear projection".

```python
patches.shape # (B, N, P^2 * C)
proj= nn.Linear(patches.shape[-1], D)
proj(patches).shape, pos_emb.shape #(B, N, D), (N, D)
x = proj(patches) + pos_emb #works with no error
```

### Class token

A question that you may have is, if transformers were originally used to generate the next item in a sequence, how would we use a transformer to perform a classification task?

The answer used in the paper is that the authors prepend a learnable embedding to the beginning of the embedded patches. The idea is that eventually after the embeddings pass through the transformer, the model uses a "classification head" to interpret the class token and output the final class.

We again implement this through a lookup table, except this time there is only one possible input which retrieves the learned embedding. 



This table looks something like this:

```python
#the CLASS token retrieves a D dimensional learnable embedding
self.cls_token_table = nn.Embedding(1, D)

#or the more common implementation, which is
self.cls_emb = nn.Parameter(torch.zeros(1, 1, D))
#which can then be initalized to weights in a normal distribution as
nn.init.normal_(self.cls_emb, std = 1e-6)
```

And in the forward function of our model, we can call the following to encode our patch sequence:

```python
cls_emb = self.position_embedding_table(torch.arange(1))

#or just self.cls_emb if using the nn.Parameter implementation.
```



Let's implement a preliminary version of the model wrapper class for the vision transformer with patches and position embeddings included.

```python
class ViT(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        #patch embedding
        self.patch_embedding = lambda x: x.reshape(B, N, P**2*C)
        #extra position (N ---> N+1) because of CLASS token
        self.pos_emb = nn.Parameter(torch.zeros(1, N+1, D)) 
        nn.init.normal_(self.pos_emb, std = 1e-6)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.cls_emb, std = 1e-6)
        self.proj = nn.Linear(P**2*C, D)
        
    def forward(self, idx, targets):
        '''
        idx and targets are the input and target blocks
        respectively that we get from the dataloader
        '''
    
        patches = self.patch_embedding(idx) #(B, N, P^2*C)
        #expands the embedding to have a batch dim
        cls_emb = self.cls_emb.expand(B, -1, -1) #(B, 1, 1)

        #project patches to token dimension
        patches = self.proj(patches) #(B, N+1, D)

        #concatenate the class embedding to the front of the patches in the N dimension
        x = torch.cat((cls_emb, patches), dim = 1) #(B, N+1, D)

        #add the position embedding to keep positional info
        x = x + self.pos_emb
```




