# Multimodal/Contrastive Learning, or CLIP from Scratch

Now that we've gone over text transformers and vision transformers, I wanted to move on to CLIP which is text-image pairs. I will be referencing Learning Transferable Visual Models From Natural Language Supervision by Radford et al. 2021.

Let's get started!

## Intro and Motivating Work

So, in the last post I talked briefly about how the training for the ViT was done using ImageNet and other labelled datasets. 
The authors of this paper point out that in NLP, pretraining data has shifted from these kinds of high-quality crowd-labeled datasets 
to using web-scale collections of text. The intuition then is that computer vision could also 
take advantage of this large amount of available text to help inform the task of image classification. 
Specifically, the authors mention the potential of "using natural language supervision for image representation learning",
which was still rare at the time of publication. CLIP, or Contrastive Language-Image Pre-training, seeks to test this
method of natural language supervision for image learning. 

The authors find that this method of changing the learning objective from predicting a certain class to a contrastive objective of finding which text-image pairs match has better rates of transfer to different datasets. 

<img width="70%" alt="image" src="https://github.com/user-attachments/assets/4352fce4-2b02-4576-a974-e6c696e73596">

## Overall Approach

### Data 

So the authors of this paper had to curate their own dataset called WebImageText which is created by searching for 
image, text pairs from a list of 500,000 queries. They mention that the resulting dataset has a similar total word count as the dataset for GPT-2.

### Training method

So, this paper makes a key innovation in the pretraining stage which allows for much better performance.
In previous works with natural language supervision, the text transformer is trained to predict the exact caption of an image. 
However, since this task of captioning is pretty complex, the authors found they had issues with scaling this approach.

Instead, they use a proxy task of predicting which text as a whole is paired with which image, or changed the predictive objective to the contrastive objective. 

In other words, "given a batch of N (image, text) pairs, CLIP is trained to predict which of the NxN possible (image, text) pairings across a batch actually occurred".
This is done by learning a multi-model embedding space, or an embedding space which expresses properties of both the text and the image.
This space is obtained by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings which are actually paired and minimize the cosine similarity between the image and text embeddings which are not paired.

Basically, if we have the following image, text pairs:

(cat, image of cat)

(dog, image of dog)

(bear, image of bear)

We want to maximize the cosine similarity between the embeddings of cat, image of cat and minimize the cosine similarity between dog, image of cat and bear, image of cat, and similarity for the other images.

#### Cosine similarity

The cosine similarity between two embeddings is defined as the cosine between the two vectors. We can use the formula, 
|a|x|b|xcos(theta) = a dot product b and rearrange it to get

<img src = "https://miro.medium.com/v2/resize:fit:1400/1*LfW66-WsYkFqWc4XYJbEJg.png" alt = "cosine similarity" width = "80%"/>

Intuitively, if we think of each value in the embedding vector as an expression of a property, then closer embeddings will have a smaller cosine between the vectors in high-dimensional space.

So let's say that we have 3-dimensional embeddings--- in this simplified example, the first dimension
could be how domesticated the thing is, the second could be how big the thing is, and the third could be how loyal the thing is.

So, if we express cat, dog, and bear in this system, we could get something like the following embeddings:


cat: <0.8, -0.2, -0.3>

dog: <0.9, 0.5, 0.7>

bear: <-0.8, 0.8, 0.5>

The embedding between cat and dog would have a cosine similarity of 0.29, and bear and dog would have a cosine similarity of 0.077.
Thus, we would say that cat and dog are more similar than bear and dog.

In the case of CLIP, we not only have embeddings for the text descriptions but also the images, and then try to match them with cosine similarity.

### Architecture

The general architecture of the model is that there is a text encoder which encodes natural language description and an image encoder which encodes an image. The authors then use a linear projection to map the encoder representation to a multi-modal embedding space. 

The authors try two architectures for the image encoder, the ResNet architecture and the ViT, which was newly introduced at the time.
The text encoder is a text transformer, which uses masked self-attention because the authors wanted to allow for initializing the transformer with pretrained models (which are decoder-only).

They find that the best performing architectures were using an image transformer and text transformer.

Thus, because we already went over how to make an image transformer and text transformer, the implementation of this model is relatively straightforward.


## Implementation


### Image and Text encoders

This section will just be mostly reusing code that we already went through when making a text and vision transformer.

Let's start with the building blocks of both transformers, which are attention, multi-headed attention, MLP, and blocks.





