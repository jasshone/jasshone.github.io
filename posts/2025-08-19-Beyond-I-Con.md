# Beyond I-Con: A Roadmap for Representation Learning Loss Discovery

*Jasmine Shone\*, Zhening Li\*; in collaboration with Shaden Alshammari, Mark Hamilton, Bill Freeman*  
*MIT CSAIL*  
*\*Equal contribution*

## TLDR
A previous paper, I-Con, created a framework showing that most representation learning methods can be explained as minimizing KL divergence between a data distribution and a learned distribution encoding similarities between data points. But why KL specifically? Properties of KL divergence such as asymmetry and unboundedness may create optimization challenges, and a KL-based loss may be misaligned with the true objective. In this work, we generalize I-Con by replacing KL with alternative f-divergences, systematically discovering novel loss functions. Our key findings: (1) Total Variation distance achieves state-of-the-art unsupervised clustering on ImageNet-1K; (2) Jensen-Shannon divergence outperforms KL for supervised contrastive learning on CIFAR-10; (3) bounded f-divergences resolve SNE's known crowding problem and produce better-separated dimensionality reduction visualizations. We also find evidence that KL's unbounded gradients cause training instability that bounded divergences avoid.

## Introduction

Representation learning has seen rapid progress through contrastive, clustering, and generative objectives, yet most methods implicitly optimize a single dissimilarity measure—the Kullback–Leibler (KL) divergence. The recent *Information Contrastive* (I-Con) framework elegantly unified over 23 representation losses under integrated KL minimization. But if representation learning methods can be unified under minimizing the divergence between two distributions, what happens when we systematically explore alternative divergences?

We present **Beyond I-Con**, a framework that enables systematic discovery of novel loss functions by exploring alternative statistical divergences in place of KL.

## Background: The I-Con Framework

The I-Con framework unifies representation learning methods by framing them as minimizing the average KL divergence between two conditional "neighborhood distributions" that define transition probabilities between data points. A fixed "supervisory" distribution p(j|i) is derived from the dataset, and a learnable distribution q(j|i) is computed from similarities between learned features. The I-Con loss is:

$$\mathcal{L}_{\text{I-Con}} = \mathbb{E}_{i \sim p(i)} \left[ D_{KL}\left( p(\cdot|i) \| q(\cdot|i) \right) \right]$$

By varying how p is constructed from the dataset and how q is defined in terms of feature similarities, this single formulation reproduces the loss functions of many existing representation learning methods.

## Beyond I-Con: Replacing KL with Alternative Divergences

Our key insight is simple: replace the KL divergence with any positive-definite divergence D:

$$\mathcal{L}_{\text{Beyond I-Con}} = \mathbb{E}_{i \sim p(i)} \left[ D\left( p(\cdot|i) \| q(\cdot|i) \right) \right]$$

We focus on **f-divergences** because they are most directly comparable to KL as measures of distance between distributions. Specifically, we explore:

- **KL Divergence**: The standard choice; asymmetric and unbounded.
- **Total Variation (TV)**: Based on the L₁ norm of probability differences; bounded between 0 and 1.
- **Jensen-Shannon Divergence (JSD)**: A symmetric, bounded variant that directly remedies KL's asymmetry and unboundedness.
- **Hellinger Distance**: Another bounded, symmetric f-divergence.

Some of these divergences—such as JSD—directly address known weaknesses of KL like asymmetry and unboundedness, making them theoretically motivated alternatives.

## Experimental Results

We evaluated these divergences across three core representation learning tasks: unsupervised clustering, supervised contrastive learning, and dimensionality reduction.

### Unsupervised Clustering on ImageNet-1K

We modified the Pointwise Mutual Information (PMI) clustering algorithm to use different divergences, following the same training setup as the I-Con paper: clustering DINO ViT embeddings on ImageNet-1K by training a linear classifier for 30 epochs with batch size 4096, learning rate 1×10⁻³, and the Adam optimizer.

| Method | DiNO ViT-S/14 | DiNO ViT-B/14 | DiNO ViT-L/14 |
|--------|---------------|---------------|---------------|
| k-Means | 51.84 | 52.26 | 53.36 |
| TEMI | 56.84 | 58.62 | — |
| Debiased InfoNCE (Previous SOTA) | **57.8** ± 0.26 | 64.75 ± 0.18 | 67.52 ± 0.28 |
| JSD | 53.50 | 63.80 | 66.60 |
| **Total Variation (Ours)** | 55.90 | **65.13** ± 0.13 | **68.40** ± 0.29 |
| Hellinger | 54.90 | 63.80 | 67.85 |

*Hungarian Accuracy on ImageNet-1K clustering.*

Total Variation outperforms the previous state-of-the-art on ViT-B/14 and ViT-L/14 embeddings.

### Supervised Contrastive Learning on CIFAR-10

We trained ResNet-50 models with supervised contrastive learning on CIFAR-10, using a Euclidean distance metric on features. Models were trained for 150 epochs with batch size 2048 and learning rate 1×10⁻³, and we systematically varied the divergence measure. Classification was performed by training a linear probe or applying k-nearest neighbors.

| Divergence | Linear Probe Acc. | k-NN (k=7) Acc. |
|------------|-------------------|------------------|
| KL | 90.03 ± 0.14 | 89.61 ± 0.13 |
| TV | 83.23 ± 0.18 | 82.95 ± 0.16 |
| Hellinger | 90.47 ± 0.08 | 90.40 ± 0.09 |
| **JSD** | **90.84** ± 0.11 | **90.62** ± 0.11 |

*Downstream classification accuracy from supervised contrastive features on CIFAR-10. Errors are standard errors of the mean over 5 seeds.*

Both Hellinger and Jensen-Shannon divergence outperform vanilla supervised contrastive learning (which uses KL), with JSD achieving the best performance.

### Dimensionality Reduction on CIFAR-10

We ran SNE with a CNN backbone on CIFAR-10 using different divergences. The qualitative differences are striking: while SNE with KL produces highly overlapping clusters, the other divergences achieve much cleaner class separation. This directly addresses SNE's well-known "crowding problem."

## Analysis and Discussion

### Why Does KL Underperform?

Across all three tasks—unsupervised clustering, supervised contrastive learning, and dimensionality reduction—a non-KL divergence outperforms KL. We hypothesize this stems from KL's unbounded penalty: when q(j|i) → 0, KL diverges to infinity. This means the loss overly penalizes placing dissimilar points far apart in the feature space, which causes different clusters or classes to crowd together and start overlapping.

This is precisely the well-known crowding problem in dimensionality reduction. When mapping clusters from a high-dimensional space to a lower-dimensional one, maintaining minimum separation distances forces some cluster pairs to become very distant (as established by packing arguments from Rogers, 1964). KL heavily penalizes these cases where q(j|i) ≪ p(j|i). The optimization responds by pulling distant clusters closer together—creating overcrowding.

TV, JSD, and Hellinger all remain bounded as q(j|i) → 0, so they incur a much lower penalty for far-apart clusters, resolving the crowding issue. Our SNE visualizations confirm this: bounded divergences produce well-separated clusters while KL does not.

### Gradient Instability with KL

We also observe evidence that KL-based losses produce unstable gradients during training, consistent with previous findings in other domains. Our gradient norm plots during SNE training show large spikes when using KL, especially near the beginning of training. In contrast, bounded divergences (TV, Hellinger, JSD) exhibit more stable gradient behavior throughout training across all network layers.

## Limitations and Future Directions

While our framework successfully demonstrates that alternative divergences can outperform KL, several avenues remain open.

We focused on f-divergences as the most natural generalization. Other divergence families—such as Wasserstein distance or integral probability metrics—remain unexplored and could offer additional benefits, particularly in terms of geometry-awareness, though computational overhead is a practical concern.

The theoretical understanding of *why* specific divergences perform best for specific tasks remains incomplete. A deeper analysis of optimization landscapes and gradient dynamics across different divergence choices would strengthen the framework.

Finally, the Beyond I-Con framework can be applied to any existing loss function captured by I-Con. We demonstrated improvements on PMI clustering, supervised contrastive learning, and SNE, but the same divergence-swapping approach could be applied to methods like SimCLR, CLIP, and others.

## Conclusion

Beyond I-Con challenges the default reliance on KL divergence in representation learning by showing that alternative f-divergences can yield superior performance across unsupervised clustering, supervised contrastive learning, and dimensionality reduction. By extending the I-Con framework with a new dimension—the choice of divergence—we provide a systematic approach for discovering novel loss functions. Our results highlight that this design choice matters and should be carefully considered rather than defaulted to KL.