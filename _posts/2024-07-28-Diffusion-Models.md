# Diffusion Models

My original plans were to make a post about some papers in RL first, but I got busy this week and the need to understand diffusion models got more relevant for my research. Therefore, I will be talking about Diffusion Models instead in this post.

I will first give a high-level overview of diffusion models based on some blogs, explainers, and Denoising Diffusion Probabilistic Models by Ho et al., then try to implement a diffusion model from scratch as per the other posts.

Let's get started!

# Core Ideas of Diffusion Models

Diffusion models try to generate data similar to their training data. The way they do this is through cleverly adding Gaussian noise to the image and then learning to recover the original image from this noise.
Then, after training, it can take in random noise and create data. To further explain this process, we have to delve into the realm of math (specifically Markov Chains and probability).

## Math, or Markov Chains and Probability

Markov chains are a fancy name for a relatively intuitive idea. Say that if it's sunny today, there is a 20% chance that it'll be rainy tomorrow, 10% chance it'll be cloudy, and 70% chance it'll continue being sunny. Say that we similarly know exactly the probabilities that we will reach a kind of day given that we start with a rainy and cloudy day.

This can be modeled with a Markov Chain, with arrows out representing moving to another state and arrows in representing moving to that state.

<img width="50%" alt="markov chain" src="https://github.com/user-attachments/assets/b8ad340a-d8d9-4acf-84c6-8e7d0ed13f89">

Markov Chains are specifically processes where the probability of transitioning to any state depends only on the current timestep and the current state. In diffusion models, that would mean that noising an image only depends on the current image. 



The pure noise image at the final noising timestep is counted as the `posterior`, which can be better explained by looking at Bayes Rule.

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/929cf2d6-ecb3-4521-bcda-8da9e8d3cd1d">

You may have seen Bayes formula before, but in simple terms, the probability of `A` being true given that `B` is true is equal to the probability of `B` being true * probability of `A` being true divided by the probability of `B`.
It can be derived from the fact that the probability that `A` is true given `B` is the probability of `A` and `B` is true over the probability of `B` being true, since the probability of `A` and `B` being true is equal to `P(B|A)*P(A)` (`A` happening and then the probability that `B` happens given that is true).

The goal of a diffusion model is to denoise noise to obtain the original input. In terms of Bayes Rule, that means that we want to know the probability of `A`, the original input, given the noise `B`. However, it's a bit easier than that, since we can just try to learn each individual step of denoising.

To formalize this a bit more, we can call the approximate posterior of this model $`q(x_{1:T}|x_0)`$ where $`x_1,...x_T`$ are the transformed input at each noising timestep and `x0` is the original input.
We then want to learn $`p_\theta(x_{t-1} | x_t)`$, or the probability that the image before the most recent noising episode was $`x_{t-1}`$ given that we currently have transformed input $`x_{t}`$.

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/9b8ab9df-f16f-4e5d-9f43-b6d10f30f7e8">

Noising


<img width="90%" alt="image" src="https://github.com/user-attachments/assets/0543703e-e33f-4864-b91d-7bc136954848">

Denoising

## Even More Math, or Gaussian Noise

So the forward process can be formally stated as follows:

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/1e7dc3e7-4a2a-4c14-9d47-42a7cf45a68b">

This comes from the statement that adding Gaussian noise to an image is equivalent to sampling from a Gaussian distribution whose mean is the previous value/image in the chain.

Here's a good proof from [this blog](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/):

We want to prove that 

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/0aa95df7-4b7b-4cef-9fbf-33b055738a81">

And the article uses a slight abuse of notation as follows: 

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/a4b6b0de-f4b0-437c-b13d-d7fd79a9cd8b">

Or specifically denoting the noised distribution of $x$ with mean $\mu$ and standard deviation $\sigma$.

<img width="80%" alt="image" src="https://github.com/user-attachments/assets/e49ef22b-d44d-4cd3-ada3-0d8f62fdf86b">

To understand this proof, you would need some background in probability. I think the general idea of why we need this proof and why it makes sense is that in the diffusion model training process, we want to not have to add noise step by step and skip straight to the current timestep t, and have the model predict the noise difference between that noised image and the original image. This works fine for training because of the property of the sum of two normal distributions being another normal distribution, so we can approximate the effect of t steps of adding noise by one gaussian distribution with the same mean and standard deviation as the sum of the guassian noise added per step up to step t. 

The actual paper has some additional parameters, specifically beta, beta tilde, alpha, alpha bar, etc. These are chosen contingent on beta, which is the amount of noise to add at each step. We will talk about these more later when we actually implement the model.

## Training the Model (+ some more math)

The model we will use is called a UNet, which looks as follows:

<img width="80%" alt="unet" src="https://github.com/user-attachments/assets/af0e8a78-169f-4d1e-8ec2-8753c200af3c">

The UNet was originally used for image segmentation, but its ability to do image processing, extract useful features from an input and provide an output size identical to the input is desirable for our task as well.

To train the model, for each image in the dataset, we want to randomly sample a timestep and compute the diffusion process for that image, after which we obtain a noisy image and the actual noise used. We will then get the model to predict the noise added to the image.

In terms of loss, we want to use the KL Divergence. The KL Divergence measures the difference between two distributions, where a lower KL Divergence meaning that two distributions are more similar. This makes sense for our problem, because we want to minimize the divergence between the probability distribution of obtaining a denoised image given a noised image and the predicted probability of doing so.

This can be approximated as a L2 loss function (the details of which are pretty complex, so they are omitted).

# Making a Minimal Diffusion Model

So I will be attempting to create a minimal diffusion model, referencing various resources and a bit of the original paper (which is quite dense, but I'll try my best).
One repo I found particularly helpful was [this repo by Dominic Rampas](https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py), as well as [this blog](https://medium.com/@mickael.boillaud/denoising-diffusion-model-from-scratch-using-pytorch-658805d293b4)

## Part 0: Notation

First, let's go over some important variables and their meanings. 
$\beta_t$ or beta at timestep t: the amount of noise to add at the given timestep, determined by a scheduler.

$\alpha_t$ or alpha at timestep t: $1-\beta_t$

$\bar{\alpha}_t$, or alpha hat: product of all alpha values from the first timestep to timestep `t`

## Part 1: Noise

A fundamental part of a diffusion model is the noising process. To do this, we need to get the noise scheduler, $\beta$, and then apply the result to an input.

For simplicity of implementation, we will be using a linear noise scheduler, defined as follows:

```python
def schedule_noise(beta_start, beta_end, steps):
  '''
  returns a linear schedule from `beta_start` to `beta_end`, with `steps` steps in between
  '''
  return torch.linspace(beta_start, beta_end, steps)

```
Then, we can calculate the alpha and alpha hat values for each timestep as follows:

```python
beta = schedule_noise(beta_start, beta_end, steps)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
```

Now that we have these values, we can get the noised images at a specific timestep given input `x` by rearranging this equation:

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/fab122f3-045d-439d-a890-a87dadf639f5">

to get

$x_t = \sqrt{\bar{a}_t} * x_0 + \sqrt{1-\bar{a}_t}*\mathcal{E}$ where $`\mathcal{E}`$ is the gaussian noise.


Let's implement this in code.

```python
def get_noised_images(x, t, alpha_hat):
  epsilon = torch.rand_like(x) #B C H W
  sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).view(-1, 1, 1, 1) #B 1 1 1
  sqrt_one_minus_alpha_hat = torch.sqrt(1-alpha_hat[t]).view(-1, 1, 1, 1) #B 1 1 1
  return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
  
```

To validate that this works for yourself, you can run this code snippet which is the previous code snippets composed with random parameter values.

```python
import torch
def schedule_noise(beta_start, beta_end, steps):
  '''
  returns a linear schedule from `beta_start` to `beta_end`, with `steps` steps in between
  '''
  return torch.linspace(beta_start, beta_end, steps)

beta = schedule_noise(1e-4, 0.02, 10000)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
def get_noised_images(x, t, alpha_hat):
  epsilon = torch.rand_like(x) #B C H W
  sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).view(-1, 1, 1, 1) #B 1 1 1
  sqrt_one_minus_alpha_hat = torch.sqrt(1-alpha_hat[t]).view(-1, 1, 1, 1) #B 1 1 1
  return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

x_t = get_noised_images(torch.rand((8, 1, 32, 32)), 1000, alpha_hat)
print(x_t[0].shape, x_t[1].shape)
```

This should print `torch.Size([8, 1, 32, 32]) torch.Size([8, 1, 32, 32])`, which is the shape of the noised image and the noise applied to the original respectively.

The function to generate the noise is now completed, and now we can move on to the model.

## Part 2: Model

The model that we will be using is a UNet, which again has the following architecture:

<img width="80%" alt="unet" src="https://github.com/user-attachments/assets/af0e8a78-169f-4d1e-8ec2-8753c200af3c">

Let's first implement the various blocks that make up this architecture, then put it all together. 

### 2a. Double Convolution
These are the various blocks that appear through the architecture, such as these:
<img width="86" alt="image" src="https://github.com/user-attachments/assets/dab60d94-4ca1-4433-93fa-1d3cec6072c5">

<img width="86" alt="image" src="https://github.com/user-attachments/assets/210ae99e-5bc5-41be-b04f-ee9ad7400cd9">

There are two convolutional layers with kernel size 3 and an activation between them. While the original paper uses ReLU, GeLU is now also used. 
Here's some accompanying code for this block.

```python
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels,mult = 2, residual = False):
    super().__init__()
    #pre-norm
    self.net = nn.Sequential(
      nn.GroupNorm(1, in_channels),
      nn.Conv2d(in_channels, mult*out_channels, 3, padding = 1),
      nn.GELU(),
      nn.GroupNorm(1, out_channels*mult),
      nn.Conv2d(out_channels*mult, out_channels, 3, padding = 1)
    )
    #specifying if there's a residual stream
    self.residual = residual
  def forward(self, x):
    if not self.residual: return self.next(x)
    return F.gelu(x + self.net(x))
```

### 2b. Down Block
Another block that is needed is the `Down` blocks, which are the downward arrows in the picture of the UNet architecture. These consist of a max pooling layer and two double convolutional layers. After the input passed through this block, we add an additional timestep embedding which consists of a SiLU activation and a linear layer. SiLU is another kind of activation has been shown to improve the performance of deep learning models over ReLU.


Here's the code for a `Down` block.

```python
class Down(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim = 256):
    super().__init__()
    self.block = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_channels, in_channels, residual = True),
      DoubleConv(in_channels, out_channels),
    )
    self.pos_emb = nn.Sequential(
      nn.SiLU,
      nn.Linear(
        emb_dim,
        out_channels
      ),
    )

  def forward(self, x, t):
    x = self.block(x)
    t_emb = self.pos_emb(t).view(-1, -1, 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1]) #reshape to be compatible with adding to x
    return x + t_emb
```

### 2c. Up Block
Another block that is needed is the `Up` blocks, which are the upward arrows in the picture of the UNet architecture. These consist of an upsampling layer and two double convolution layers. After the input is upsampled, we concatenate the value of x in the corresponding denoising timestep, which is called a skip connection, before passing the vector through the convolution layers. Then, we again add an additional timestep embedding which consists of a SiLU activation and a linear layer. 

Here's the code for a `Up` block.

```python
class Up(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim = 256):
    super().__init__()
    self.up = nn.Upsample(scale_factor = 2, mode = "nearest", align_corners = True)
    self.block = nn.Sequential(
      DoubleConv(in_channels, in_channels, residual = True),
      DoubleConv(in_channels, out_channels, 1/2),
    )
    self.pos_emb = nn.Sequential(
      nn.SiLU,
      nn.Linear(
        emb_dim,
        out_channels
      ),
    )

  def forward(self, x, skip_x, t):
    x = self.up(x)
    x = self.block(torch.cat([skip_x, x], dim = 1))
    t_emb = self.pos_emb(t).view(-1, -1, 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1]) #reshape to be compatible with adding to x
    return x + t_emb
```
### 2d. Putting it all together

Now, we just need to compose these blocks into a UNet architecture.

This is done following this image again:

<img width="80%" alt="unet" src="https://github.com/user-attachments/assets/af0e8a78-169f-4d1e-8ec2-8753c200af3c">

Here's the code, which isn't too interesting besides replicating the architecture.

```python
class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = DoubleConv(1, 64)
    self.down1 = Down(64, 64)
    self.conv2 = DoubleConv(64, 128)
    self.down2 = Down(128, 128)
    self.conv3 = DoubleConv(128, 256)
    self.down3 = Down(256, 256)
    self.conv4 = DoubleConv(256, 512)
    self.down4 = Down(512, 512)
    self.conv5 = DoubleConv(512, 512)
    self.up1 = Up(1024, 512)
    self.conv6 = DoubleConv(512, 256)
    self.up2 = Up(512, 256)
    self.conv7 = DoubleConv(256, 128)
    self.up3 = Up(256, 128)
    self.conv8 = DoubleConv(128, 64)
    self.up4 = Up(128, 64)
    self.out = nn.Conv2d(64, 1, kernel_size = 1)
  
  def forward(self, x, t):
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, 256)
    x1 = self.conv1(x)
    x2 = self.down1(x1, t)
    x3 = self.conv2(x2)
    x4 = self.down2(x3, t)
    x5 = self.conv3(x4)
    x6 = self.down3(x5, t)
    x7 = self.conv4(x6)
    x8 = self.down4(x7, t)
    x9 = self.conv5(x8)
    x10 = self.up1(x9, x7, t)
    x11 = self.conv6(x10)
    x12 = self.up2(x11, x5, t)
    x13 = self.conv7(x12)
    x14 = self.up3(x13, x3, t)
    x15 = self.conv8(x14)
    x16 = self.up4(x15, x1, t)
    
    return self.out(x16)
```

### 2e. Adding a sinusoidal timestep position encoding

One final thing we can do to improve performance is to add a sinusoidal timestep position encoding before passing it into the model. I won't delve too much into this here because it isn't the focus of the article, but here is a function which does this from [this repo which I linked before](https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py#L128):

```python
def pos_encoding(self, t, channels):
  inv_freq = 1.0 / (
      10000
      ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
  )
  pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
  pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
  pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
  return pos_enc
```
Integrating this into our implementation, we get:

```python
class UNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = DoubleConv(1, 64)
    self.down1 = Down(64, 64)
    self.conv2 = DoubleConv(64, 128)
    self.down2 = Down(128, 128)
    self.conv3 = DoubleConv(128, 256)
    self.down3 = Down(256, 256)
    self.conv4 = DoubleConv(256, 512)
    self.down4 = Down(512, 512)
    self.conv5 = DoubleConv(512, 512)
    self.up1 = Up(1024, 512)
    self.conv6 = DoubleConv(512, 256)
    self.up2 = Up(512, 256)
    self.conv7 = DoubleConv(256, 128)
    self.up3 = Up(256, 128)
    self.conv8 = DoubleConv(128, 64)
    self.up4 = Up(128, 64)
    self.out = nn.Conv2d(64, 1, kernel_size = 1)

  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc
  def forward(self, x, t):
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, 256)
    x1 = self.conv1(x)
    x2 = self.down1(x1, t)
    x3 = self.conv2(x2)
    x4 = self.down2(x3, t)
    x5 = self.conv3(x4)
    x6 = self.down3(x5, t)
    x7 = self.conv4(x6)
    x8 = self.down4(x7, t)
    x9 = self.conv5(x8)
    x10 = self.up1(x9, x7, t)
    x11 = self.conv6(x10)
    x12 = self.up2(x11, x5, t)
    x13 = self.conv7(x12)
    x14 = self.up3(x13, x3, t)
    x15 = self.conv8(x14)
    x16 = self.up4(x15, x1, t)
    
    return self.out(x16)

```

Whew! That was a lot, but now we are mostly finished.

## Part 3 Training the model

Given a batch of images, the way you train the model is:

1. sample random timesteps to generate images for
2. add noise to the images to these timesteps
3. predict the noise using the UNet
4. calculate the loss

Here's how you would do this given a batch of images, `images`:

```python
model = UNet()
optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
MAX_STEPS = 10000
mse = nn.MSELoss() #l2 loss
t = torch.randint((low=1, high=MAX_STEPS), size = (images.shape[0],)) # generate B random timesteps
x_t, noise = get_noised_images(images, t)
pred = model(x_t, t)
loss = mse(noise, pred)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

To get batches of images, we can use the MNIST dataset and load it into a Dataloader.

```python
import torchvision
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

Now, with this, we can create a full training loop! (Adapted from [here](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb#scrollTo=6MW0xsLGNrXL))

```python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epochs = 3

net = UNet()
net.to(device)

loss_fn = nn.MSELoss()

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = []

for epoch in range(n_epochs):

    for x, y in train_dataloader:

        x = x.to(device)
        x = F.interpolate(x, (32, 32))
        t = torch.randint(0, 10000, (x.shape[0],)).to(device) # Random timesteps
        x_t, noise = get_noised_images(x, t, alpha_hat) 
        #print(x_t.shape)

        pred = net(x_t,t)

        loss = loss_fn(pred, x) 

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

# View the loss curve
plt.plot(losses)
plt.ylim(0, 0.1);
```

Here are the results I got from training the model on 10 epochs:

![image](https://github.com/user-attachments/assets/dd1ddb93-e068-4063-83d0-29eff887414b)

And here's a visualization of the impact of the amount of noise on the model's ability to denoise:

![image](https://github.com/user-attachments/assets/29ce738f-8328-4a76-9027-28ca536c6d4f)

As you can see,the more noise is added, the worse the model gets at trying to recover the original image. 

# Final Thoughts

Diffusion models have become a lot more well known after the introduction of stable diffusion as well as their use in Dalle-2. Another interesting fact is that diffusion models (specifically conditional diffusion models) are being used in policy learning in robotics for their ability to generate diverse samples based on a training distribution.

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/d031d5b2-9f25-490a-b0b5-37998e89a21c">

One advantage of diffusion models versus the other popular alternative for image synthesis, GANs, is that the training process is much more stable. Additionally, diffusion models can be conditional, which means that their generations are conditioned on some input such as text or a lower resolution image. This allows for diffusion to be used for a lot of different tasks such as inpainting, super-resolution, and text-to-image.

That's it for this article—hope you learned something, and thanks for reading!

