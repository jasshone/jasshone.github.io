# A Deep Dive Into Transformers
<img width="40%" alt="image" src="https://github.com/user-attachments/assets/df39c467-c620-489b-a167-512b41c52d64">


I went through the classic Karpathy tutorial on how to build a transformer from scratch, and I thought there was an opportunity to further digest and make explicit the concepts and elements that were covered in the tutorial in a written format. My goal is to make everything as understandable/"follow-alongable" as possible, in part also for my own understanding. Without further ado, here is my written take on making a basic language transformer. 

### Data Processing

We start by getting the input text:

```python
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

And then reading it using the python file reader.
```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

Then, the next step is to encode the text input into numbers. The way that this is handled in this particular tutorial is very straightforward: just getting a set of all chars in the text and sorting them. The vocab size is the number of these unique characters.

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

Then, we number each of these unique chars from 0 to `vocab_size`, and create a dictionary that maps char to num and num to char. 

```python

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

```

Encoding and decoding becomes equal to splitting the characters into a list and converting them to the integer equivalent for encoding and joining the corresponding chars back together for decoding.

```python

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
```

After converting the data to a list of numbers, we split it into training and validation sets by taking the first 90% to be the training set.

```python

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

```

For sequence-based data, the training method is a bit different from traditional input-output pairs. Essentially, the way that training works is that for a given chunk of data with size `block_size`, for `i` from 0 to `block_size`, we take the first `i` inputs to be the context and try to predict the `i+1`th input from that. This is illustrated clearly by this code snippet:

```python

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

```

Thus, to get a batch of data, we generate `batch_size` random indices in the data which has at least `block_size` following inputs and stack the tensors for those blocks to get the inputs and the targets. 

```python

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y

```


And that's it for the data portion! (22:28 in the video)

---
### Bigram Language Model

This part is motivated by 1. getting a baseline of performance to compare later models to and 2. introducing some concepts that are also used in later models. 

A Bigram Language Model is a simple language model which only uses the last input to predict the target. You can think of it as a lookup table of sorts which predicts the next character by knowing the current character and the probability that every other character has to follow the current character.

For example, let's say for the sake of simplication that only the characters "e", "r", "k", and "a" exist in our input data. If the current character is "e", then we could know that the frequency of "r" following "e" is 4, "k" following "e" is 1, and "a"/"e" following "e" is 0. Then, by taking the softmax, we can convert those frequencies into probabilities, and by sampling from the probability distribution the model will generate "r" as the next letter most often, and so on.

Coding/training this model is relatively simple as a result: the model just needs to store/train a lookup table between every pair of possible characters. 

To generate content from this model, we just need to use this lookup table for a starting character, convert the logits for each other character into probabilities, then sample from this probability distribution to generate the next character, then repeat until we have generated enough characters. 

Here's the corresponding code for this model:

```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets):
        '''
        idx and targets are the input and target blocks
        respectively that we get from the dataloader
        '''
    
        logits = self.token_embedding_table(idx)
        #(B, T, C) because idx is B,T and for each input char
        #we get the corresponding logits for each of the C possible chars
        #in our vocabulary
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #get predictions for current block
            logits, loss = self(idx)
            #get only the last time step because we want to add onto that
            logits = logits[:, -1, :]
            #get the probabilities of each possible char in the vocab
            probs = F.softmax(logits, dim = -1)
            #sample from that distribution to get the char we generate
            idx_next = torch.multinomial(probs, num_samples = 1)
            #append the index to the sequence so we can build onto it
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

Now, we just go through the standard training/val loop to train this model.

```python

m = BigramLanguageModel(vocab_size)
#used because converges faster in some situations
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(10000):
    #get train data using our batch function
    xb, yb = get_batch('train')
    #eval loss and update model
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
```

To get a generated output we just combine the decode function we coded before with the generate function of the model. The char we start with in our previous indices is set to 0 to prompt the model to start its generation.

```python
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.log), max_new_tokens = 500)[0].tolist()))
```
This will print out the generated text, which is not too good at this stage (2.5).

That's all for the Bigram Model section! (37:49 in the video)

---
### Leveraging Context

The weakness of the Bigram Model is that there's a lot more context that could be used to generate the next character than just the character before it. The next couple sections build up to the concept of attention, which is a very key component of the transformer architecture and their performance.


#### Version 1: averaging using for loop
One easy way to aggregate the information of the tokens before the target token is simply to average the logits. This can be done using a for loop. Here's some code which does this for a random input x.

```python
torch.manual_seed(1337)
B,T,C = 4,8,2 #batch, time, channels
x = torch.randn(B,T,C)

#xbow[b,t] is the average of x[b] from times 0 -> t inclusive
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t,C)
        xbow[b,t] = torch.mean(xprev, 0)
```




#### Version 2: using matrix multiply

However, the previous method is very inefficient and can be sped up using matrix multiplication.

The way that we can do this is to recognize that we can use a matrix as a mask which has 1's when a token is included in the context aggregation and 0's when it is not, and multiply it with the input. This can be done using
```python
torch.tril(torch.ones(3,3))
```

which looks like 

1, 0, 0

1, 1, 0

1, 1, 1

(notice that first, only the first token is included, then only the first and second token, then all of the tokens)

Then, to get the average, we can just normalize each row of the mask. We then get something like:

1, 0, 0

1/2, 1/2, 0

1/3, 1/3, 1/3


Here's an example of how to do this for a random vector `b`:

```python
torch.manual_seed(42)
#initialize mask
a = torch.tril(torch.ones(3,3))
#normalize to make the resulting computation averages
a = a/torch.sum(a, 1, keepdim = True)

#this vector would be the input 
b = torch.randint(0,10,(3,2)).float()

#multiply the mask by the input to compute the averages
c = a@b

```

To convert our previous code, we just need to generate a mask with the correct shape and multiply it with our input, as follows:

```python
wei = torch.tril(torch.ones(T,T))
wei = wei/wei.sum(1, keepdim = True)
xbow2 = wei @x #(B,T,T) @ (B,T,C) -> (B,T,C)
torch.allclose(xbow, xbow2) #returns True
```

#### Version 3: Adding Softmax

This version is also essentially the same as the previous versions, except it uses softmax which lends itself to the next version (self-attention). Essentially, we take an array of zeroes, set the indices corresponding to the tokens we won't use to -inf, and use softmax to convert it to the same averaging mask we used before. Here is the code:

```python
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # when tril == 0, fill in wei as -inf
wei = F.softmax(wei, dim = -1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3) #returns True
```

What is interesting about softmax is that softmax enables us to weight each token in the averaging differently (using values that are not necessarily 0), essentially creating a weighted average. In other words, tokens can find specific other tokens "more interesting" rather than finding all parts of the past input "equally interesting."


#### Adding Linear Layer + Positional Encoding to Our Model

First, we add a linear layer to process the embeddings from the embedding table, which adds a bit more complexity to our model. Then, we also add position information about each token since position now matters in our model. This is accomplished through a position embedding table which is similar to the token embedding table we previously used.

```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #there are block_size possible values for the position of a token
        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer
    def forward(self, idx, targets):
        '''
        idx and targets are the input and target blocks
        respectively that we get from the dataloader
        '''
    
        tok_emb = self.token_embedding_table(idx)
        #encodes the position of each token using the position embedding table
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb
        logits = self.lm_head(x) #pass through linear layer first
        #before computing loss
        #(B, T, C) because idx is B,T and for each input char
        #we get the corresponding logits for each of the C possible chars
        #in our vocabulary
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #get predictions for current block
            logits, loss = self(idx)
            #get only the last time step because we want to add onto that
            logits = logits[:, -1, :]
            #get the probabilities of each possible char in the vocab
            probs = F.softmax(logits, dim = -1)
            #sample from that distribution to get the char we generate
            idx_next = torch.multinomial(probs, num_samples = 1)
            #append the index to the sequence so we can build onto it
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

#### Version 4: Self-attention!!

So this is basically the crux of the tutorial, and it builds upon the last version we made with softmax. As mentioned at the end of Version 3, we want to leverage the ability of softmax to convert logits to probabilities to weight input tokens differently rather than taking a simple average of the input tokens. 

The way that we do this is as follows:

1. Every token at each position will emit two vectors: a query and a key. The query is roughly "what am I looking for", and the key vector is roughly "what do I contain".
2. To compute affinities between tokens in a sequence, we can simply take the dot product of the keys and queries. For one token, that would be the token's query dot product with all of the keys of the other tokens in the sequence.
3. If a key and a query are aligned, then the dot product will be high, which will make the head learn more from this specific token when predicting the target.
4. We take this matrix which is the result of the dot product, remove tokens which are in the future of the current token, and then softmax the result to convert the logits to probabilities
5. Then, we convert the input to a value vector, which takes the input and adds another layer of filtering/complexity to better convey the useful information the head found from this input, and multiply it by the resulting matrix in the 4th step.

To illustrate how keys and queries work, suppose we have a vocab of "a", "b", "c", "d", "e", "f" and the number of dimensions in the key/query vectors (or `head_size`) is 3. Say that "a" has the query vector of <2, 1, 0>, "b" has the key vector of <1, 2, 0>, and "c" has the key vector of <0,0,1>. By multiplying the query by the keys, we find that the head should "pay attention" to "b" more than "c" when the last token in the sequence is "a". 

Here is the implementation of a single head of self-attention:

```python
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.rand(B,T,C) # would be the input vector

head_size = 16 # dimension of the keys and queries

#map each potential token to a `head_size` dim key
key = nn.Linear(C, head_size, bias = False)
#map each potential token to a `head_size` dim query
query = nn.Linear(C, head_size, bias = False)

#map each potential token to a `head_size` dim value
value = nn.Linear(C, head_size, bias = False)

k = key(x) #(B,T,16)
q = query(x) #(B,T,16)
wei = q @ k.transpose(-2, -1) #get dot product, (B, T, 16) @ (B,16,T) ---> (B,T,T)


#remove tokens in the future of the current timestep
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # when tril == 0, fill in wei as -inf
wei = F.softmax(wei, dim = -1)
v = value(x)
out = wei @ v
```


#### Some Notes
One note that Karpathy makes is that tokens across batches do not communicate with each other, so you can treat them as separate computations. Another note is that in an encoder block, all of the nodes can talk to each other (no mask with tril), so our head is a decoder block (nodes from the future cannot be referenced).

There's also a difference between what we implemented, which is a self-attention head, and a cross-attention head. In a self-attention head, the keys, queries, and values come from the same source; in a cross-attention head, the queries may come from x, but the keys and values may come from a seperate source such as an encoder block.

Another important step that we didn't add that is in the original "Attention is All You Need" paper is scaling the result of Q @ K.T down by `sqrt(head_size)`. This is because if the key and query matrices are unit gaussian, then because the dimension is `head_size`, the variance will also be on the order of `head_size`. Since the result of Q @ K.T will be fed into softmax, it should be fairly diffuse because otherwise when faced with extreme values, softmax will converge to one-hot vectors (which limits the information each node gets to down to basically one other node).

#### Implementing an Attention Head

Now, we take what we learned about attention and convert it to an actual `Head` class. Note that the code is essentially the exact same as the attention code just with small changes to create a key, query, and value matrix for the specific head and allow the head to reuse the tril matrix.

```python

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(vocab_size, head_size, bias = False)
        self.query = nn.Linear(vocab_size, head_size, bias = False)
        self.value = nn.Linear(vocab_size, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x, targets):
        B, T, C = x.shape
        k = self.key(idx)
        q = self.query(idx)
        v = self.value(idx)
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        out = wei @ v
        return out

```

Now, all we need to do is integrate this head into the language model class.

```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        #adding a self attention head
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb

        #feed token embeddings and position embeddings through self-attention head
        x= self.sa_head(x)

        logits = self.lm_head(x)
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens to prevent going out of scope
            idx_cond = idx[:, -block_size:]

            
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

Now, we can train the model. A couple changes to the hyperparameters:

1. learning rate decrease to 1e-3 (self-attention can't tolerate high learning rates)
2. increased iterations because learning rate is lower

The loss slightly decreases from adding the head (2.4).

That was a very long section, but luckily we are now done. (1:21:58 in the video)

---
### Multi-headed Self-attention

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/5b016719-0a47-4383-88de-d9ffeb65e650">

We previously implemented a single attention head; multi-head self-attention is having multiple of these heads, and concatenating then aggregating their results. 

Implementing a class for multi-head attention is thus pretty straightforward.

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        #generating a modulelist of `num_heads` heads
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
    def forward(self, x, targets):
        #run the input through each head and concatenate
        return torch.cat([h(x) for h in self.heads], dim = -1)

```

Then, we update our language model code to use multi-headed attention. One note is that because there are now `n` heads running in parallel, the `head_size` of each will correspondingly be smaller to get the same shape we got before of the output vector.


```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #adding a multi-headed self attention
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention, same output vector dimension as before which is `n_embd`
        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb

        #feed token embeddings and position embeddings through multi-headed self-attention
        x= self.sa_heads(x)

        logits = self.lm_head(x)
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens to prevent going out of scope
            idx_cond = idx[:, -block_size:]

            
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

By training this new model, the loss reduces a bit more (2.28).

The intuition about the reduced loss is that it helps to have more communication channels between the tokens so more information can be paid attention to and learned from. With that, multiheaded self-attention is complete. (1:24:15 in the video)

---
### Feedforward layers

Previously, the model went straight from multi-headed self attention to logits. This meant that the tokens did not have much time to "think on" what they found from the other tokens. To solve this, we add a small feedforward layer with a nonlinearity to allow for this "thinking" to occur. Essentially, self-attention allows for communication between tokens, and once the communication/data gathering has occurred, now the tokens "think" on that data independently.

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)
```

We pass the output from the multi-headed self-attention through a feedforward layer before computing the logits.


```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)

        #add feedforward layer
        self.ffwd = FeedForward(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb
        x = self.sa_heads(x)

        #feed the output of the multi-headed self attention through the feedforward layer
        x = self.ffwd(x)

        logits = self.lm_head(x)
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens to prevent going out of scope
            idx_cond = idx[:, -block_size:]

            
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```

This addition decreases loss slightly to 2.24. 

---

### Blocks

Now, what we want is to stack this component of self-attention, feedforward in sequence to allow for the model to do even more "thinking". To do this, we implement a `Block` class which consists precisely of a multi-headed attention and a feedforward layer.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        #calculate size of heads by the restriction that we want the final
        #concatenated embedding from all the heads to have C = n_embd
        head_size = n_embd//n_head

        #initialize multi-headed attention + feedforward 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

Now, we want to use several blocks within our network instead of just one.

```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #add the blocks to the model
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
        )

        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb

        #pass x through the blocks
        x = self.blocks(x)

        logits = self.lm_head(x)
        B,T,C = logits.shape

        #reshape the logits/targets to what torch expects for cross entropy
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #generate a maximum of `max_new_tokens`
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens to prevent going out of scope
            idx_cond = idx[:, -block_size:]

            
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
```
---

### Optimizations

At this point, because the neural net is starting to become pretty deep, there's some optimization issues which arise which leads performance to suffer. To solve these issues, there are two optimizations that dramatically help with the depth of the network and making it sure it remains optimizable.

#### 1. Residual Connections

The first optimization is adding residual/skip connections between nodes. The way that these work is that you take the input, pass it through the block, then add the original input to the result.

<img width="40%" alt="image" src="https://github.com/user-attachments/assets/ed441789-ae33-45bb-a6fa-e056d93aacee">

You can think of it as a residual pathway for which there's a branch off of it which performs some computation, and then is combined back into the pathway by addition. In the beginning of training, this basically allows the gradients from the supervision to directly propogate back to early layers, and the intermediate blocks only kick in over time. 

Implementing these connections are relatively simple within the Block class:

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        #add residual connection
        x = x+ self.sa(x)

        #add residual connection
        x = x+ self.ffwd(x)
        return x
```

Additionally, we introduce a projection layer within both the `MultiHeadAttention` class and FeedForward class to "project" the outputs of each back to the residual pathway.

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])

        #add projection layer
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x, targets):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        #project output
        out = self.proj(out)
        return out
```

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd), #add projection layer
        )
    def forward(self, x):
        return self.net(x)
```

#### Position-wise Feed-Forward Networks
Within the original Attention is All You Need paper, the channel size of the inner layer of the feed-forward network is multiplied by 4. This change looks like the following:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd), #scale up by 4
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd), #scale down 
        )
    def forward(self, x):
        return self.net(x)
```
The loss further decreases after training this model (2.08). However, at this network size, we are starting to see some overfitting (train loss < val loss).


#### 2. Layernorm

A related concept, Batchnorm, basically makes sure that across the batch dimension, the outputs of neurons are unit gaussian (0 mean, 1 std). Layernorm is the same thing except instead of normalizing the columns of the output we normalize the rows. For each individual example, the outputs will now be normalized.

Contrary to the original paper, it is now more common to apply layernorm before multi-head attention is done.

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

We also add a layernorm after the blocks in the language model:

```python

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #add layernorm
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) #new linear layer

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb
        x = self.blocks(x)
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


There is again a slight improvement in loss by adding the LayerNorms (2.08). With that, we move on the final part of the tutorial, scaling up the model! (1:37:42 in the video)

---

### Scaling Up the Model
We have basically all of the pieces in place at this point; now it's just cleaning up the code to allow for creating larger models.

Here's what the language model class looks like now:


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

We also add dropout right before any connection back into the residual pathway. The intuition behind using dropout is that by shutting off connections between nodes randomly, we essentially train a bunch of small, partial networks, and at test time when all of the nodes are switched on, we merge these networks into a single ensemble, which improves performance. We make this addition in the following components:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd)
            nn.Dropout(dropout) #add dropout
        )
    def forward(self, x):
        return self.net(x)
```

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

The following are the list of hyperparameters used for training this larger neural net:

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6 #each head is 64 dimensional
n_layer = 6
dropout = 0.2
```

The loss decreases again from this model quite substantially (1.49) and the output is much more similar to english. Now, the coding portion is complete! (1:42:30 in the video)

---

### Conclusions + Where to Go from Here

What we implemented was a decoder-only transformer, which is usually what the pretraining step of language models are. This is when the model learns to "babble text" on and on, which is what we get in the "generate" function of our model. 

For language models like GPT, there is an extra portion of the model called the encoder which essentially learns to encode the prompt that is fed into these language models to get a relevant output (such as within a translation task). For these models, there is an extra connection from the outputs of the encoder to the decoder through a cross attention (queries from decoder block, keys and values coming from the last encoder block).

Our pretraining step was done on a transformer with 10M parameters, on a dataset with 1 million tokens (around 300,000 tokens using the OpenAI encoding scheme which uses subwords). GPT-3 has 175B parameters and was trained on 300 billion tokens.

After pretraining, the next stage is to align the model to be an assistant/create outputs corresponding to prompts. This is done by first collecting thousands of question-answer pairs and train the model to expect a question and an answer pair.

Then, the second step is to have different raters rank responses in order of preference to train a reward model to predict the desirability of each response. The third step is to optimize the policy gradient using the PPO RL optimizer to fine tune the answer policy to score a high reward according to the reward model.

These fine-tuning steps move the model from a document completer to a question-answerer. 

That is the end of the tutorial, and this article. You can find my implementation of the transformer in this blog [here](https://github.com/jasshone/GPT). Hope this explanation made sense and you learned something; I certainly did while writing it!