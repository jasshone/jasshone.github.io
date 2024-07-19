## A Deep Dive Into Transformers

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
This will print out the generated text, which is not too good at this stage. 

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
wei = F.softmax(wei, dim = -1
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
### Version 4: Self-attention!!


