# Better Notions of Similarity for Image-image Contrastive Learning

## Introduction
I also worked on another project at the Freeman lab last semester centering on image-image contrastive learning and how to make it better. This project ended up getting scooped but I learned a lot in the process of doing it. 

## Motivation
Regular contrastive learning between image pairs simply asserts that images are similar when they are identical. Naturally this leads to issues with training because some images are more similar than others. This led to the creation of various methods like SupCon and X-sample Contrastive Learning which add additional modifiers to designating when two images are "similar".

However, one issue is that these methods still don't explain *why* two images are similar. This might be important because by not specifying why images are similar the model could be inadvertently losing information which is not present in the similarity objective (i.e. color or background when using supervised contrastive learning) in embeddings.

This is the core idea of our project, which creates text-conditioned similarity between two images. 

Here is the general architecture we were using:

![backbone](images/architecture-text-sim.png)

In essence, the idea is that we want to have some text "filter" which narrows down which aspect of similarity to focus on.

## Design

The first part of the design was to create a new dataset for our task of specifying similarity between images. This was done by sampling image pairs by class, dinov2 similarity, and random matchups within Imagenet, then using GPT to give specific similarities and differences between each pair. The goal of this was to combat there being only specific ways things were similar and lose out on other kinds of similarity (i.e. background).

Then, after this dataset was collected, we first used frozen vision and text backbones + not frozen MLPs to get the embeddings for each image, image, text triplet and then projected the vision embeddings onto the text embeddings. 

For the loss I played around with a lot of different options for how to check if the two projected image embeddings are good. 

![graph of embeddings](images/emb-graph.png)

As you can see here, though, I wasn't able to get embedding clustering which passed the sanity check of at least being roughly similar to CLIP-style training.

## Scooped
Months after I started the project, I got a Slack message from my collaborator which sent another paper just published at an ICLR workshop which does the same thing we did but in a different perspective. This paper is called Focallens.

The authors cleverly use a VQA dataset (bridges text and images), frames the problem differently ("focuses" embedding according to text prompt), and also uses a different architecture (adds a text token to the input of the transformer).

However, based on the OpenReview, seems like they didn't have a good time with finding baselines which can show their method is good which are widely accepted since most baselines are class-based to some extent such as clustering.

## Ponderings
I think this project caused me to start thinking more about compression as a viable and interesting research problem. There's actually a high amount of literature on information loss depending on the task (e.g. classification) and multimodality (cross-modality information loss/interference in CLIP for example). Because in compression you choose what to "pay attention to" as a way to have intelligence when solving problems, you naturally may become worse at solving certain kinds of problems which are not represented in the training data/learning objective. Since compression is important for learning, though, the question becomes how to preserve the information we need through the specific task objective.

## Conclusion
While getting scooped kinda sucked, I guess it is a rite of passage and also it was pretty interesting seeing what the other researchers came up with (very different from what we did). I also think I wouldn't have started being interested in representations/information loss without this project. I do think there is still some merit in the core concept of adaptive embeddings like proposed in Focallens and the question is more of how to address the evaluation/benchmark gap for specifically the issue of the loss of information important for tasks such as retrieval within embeddings.