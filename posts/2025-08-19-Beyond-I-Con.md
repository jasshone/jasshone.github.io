# Beyond I-Con: Exploring New Dimensions of Distance Measures in Representation Learning

*Jasmine Shone; project in collaboration with Shaden Alshammari, Mark Hamilton, Bill Freeman*  
*MIT Computer Science*

## TLDR
A previous paper, I-Con, created a framework which describes representation learning methods. In short, this framework states that most representation learning methods can be explained as aligning a data distribution with a learned distribution, and the exact formulations for each distribution lead to a specific loss. However, (1) the distance metric between the distributions always uses KL, which is known to have specific issues, (2) the similarity kernel which creates the data distribution by measuring how similar data samples are with each other can be changed in existing losses to create new ones (most losses use cosine similarity, but alternatives like euclidean distance are less explored). In this work, we show that trying different losses based on changing the distance metric and distance/similarity leads to state-of-the-art results. Additionally, we discover that KL divergence, while effective with angular similarity measures, may exhibit training instability when paired with distance-based similarity kernels.

## Introduction

Representation learning has seen rapid progress through contrastive, clustering, and generative objectives, yet most methods implicitly optimize a single dissimilarity measure—typically the Kullback–Leibler (KL) divergence. While the recent *Information Contrastive* (I-Con) framework elegantly unified over 23 representation losses under integrated KL minimization, KL is *metric-agnostic* and can distort latent geometry. We present **Beyond I-Con**, a divergence-agnostic generalization that faithfully marries information geometry with data geometry.

## The Problem with KL-Centric Approaches

Learning representations that simultaneously capture semantic similarity and respect the underlying geometry of data manifolds is a long-standing goal in machine learning. Recent self-supervised paradigms—contrastive, clustering-based, and generative—can all be interpreted as matching conditional neighbourhood distributions. The Information Contrastive (I-Con) framework formalized this view through integrated KL divergence:

![equation 1](images/eqicon1.png)

However, KL measures mismatch *in probability space* and is oblivious to Euclidean or geodesic distances between samples. Two disjoint clusters swapped in latent space can have low KL divergence yet be geometrically disastrous for downstream tasks.

## Beyond I-Con: A Divergence-Agnostic Framework

We ask: *Can a single, geometry-aware objective subsume existing SSL losses while guaranteeing meaningful latent distances?* 

Our key insight is that I-Con represents just one point in a much larger space of possible divergence measures. Let $\mathcal{D}$ be any positive definite divergence. We define our general objective as:

![equation 2](images/eqicon2.png)

This simple substitution opens up a rich family of objectives. Choosing different values of $\mathcal{D}$ instantiates familiar objectives like InfoNCE (Jensen-Shannon divergence) and triplet loss (Total Variation).

### Divergence Families We Explore

**f-divergences**: Given convex $f$ with $f(1)=0$, these include KL, Jensen–Shannon (JSD), Total Variation (TV), and Hellinger distances. Each captures different aspects of distributional mismatch.

**Integral Probability Metrics (IPMs)**: These include Maximum Mean Discrepancy, which are inherently geometry-aware.

**Bregman Divergences**: For strictly convex $\psi$, these include Euclidean distance and Itakura–Saito divergence as special cases.


## Exploring New Loss Function Combinations

An important observation emerged from analyzing how different divergences interact with different similarity measures in the embedding space. When we systematically vary both the divergence measure (KL, Total Variation, Hellinger, etc.) and the similarity measure (angular vs. distance-based), we discover new loss function combinations that were previously unexplored.

**Angular-based methods**: These utilize dot product or cosine similarity, learning representations where similar features have high cosine similarity (small angles between vectors). This approach is common in methods like SimCLR and InfoNCE.

**Distance-based methods**: These rely on Euclidean distance in the embedding space, pulling similar instances closer together while pushing dissimilar instances apart using distance metrics. This formulation creates natural clustering structures where similar samples are geometrically proximate.

By exploring the cross-product of different divergences with different similarity measures, we uncover loss functions that had not been systematically studied. Some of these combinations, particularly Total Variation with distance-based similarity, achieve strong empirical performance.

## Experimental Results

We conducted extensive experiments across multiple domains: unsupervised clustering, supervised contrastive learning, and dimensionality reduction.

### Unsupervised Clustering on ImageNet-1K

Using DINO ViT embeddings, we evaluated different divergences and observed competitive performance from Total Variation:

| Method | DiNO ViT-S/14 | DiNO ViT-B/14 | DiNO ViT-L/14 |
|--------|---------------|---------------|---------------|
| k-Means | 51.84 | 52.26 | 53.36 |
| Debiased InfoNCE (Previous SOTA) | **57.8** | 64.75 | 67.52 |
| **Total Variation (Ours)** | 55.90 | **65.13** | **68.40** |

### Supervised Contrastive Learning

Our experiments on CIFAR-10 explored both angular-based and distance-based formulations across different divergences. We observe interesting patterns in how different divergences perform depending on the underlying similarity measure:

| Divergence | Angular-based | Distance-based |
|------------|---------------|----------------|
| KL | 92.72 / 91.33 | 57.36 / 50.55 |
| Total Variation | 85.04 / 81.80 | **96.41** / **97.33** |
| Hellinger | 89.16 / 87.12 | 92.50 / 91.93 |
| Jensen-Shannon | 86.69 / 84.03 | 91.83 / 90.99 |

*Results shown as Linear Probing / KNN accuracy percentages. All models are trained for 150 epochs with a ResNet-50 architecture.*

Notably, while KL divergence performs well with angular similarity measures, Total Variation achieves the highest performance when combined with distance-based measures. However, these final performance numbers only tell part of the story—a deeper analysis reveals differences in training dynamics across divergence-similarity combinations.

### Training Instability in Certain Divergence/Similarity Combinations

Through monitoring training dynamics, we discovered that certain divergence-similarity combinations suffer from optimization instability. Most strikingly, while vanilla supcon (KL + cosine similarity kernel) trains in a stable manner, KL divergence paired with distance-based similarity measures exhibits training collapse despite initially promising performance.

![training instability](images/instability.png)

*Training dynamics comparison: Total Variation + Distance (green) maintains stable learning throughout training, while KL Divergence + Distance (red) suffers collapse. Though this example shows collapse around step 1200, we observed similar patterns across multiple runs, though timing varied.*

The training curves reveal that KL divergence with distance-based similarity initially learns effectively, often tracking the performance of Total Variation (the best method when utilizing a distance-based similarity kernel) and reaching validation accuracies near 80%. However, the optimization becomes unstable at unpredictable points during training, leading to crashes that drive performance down to 10-20% accuracy. 

This instability may be intrinsic to the divergence-similarity combination. Across multiple independent runs, we observe collapses for KL divergence with distance-based similarity, though the exact timing varies. In contrast, Total Variation with distance-based similarity maintains stable training throughout, smoothly converging to high performance without any observed instability across all experimental runs.

## Key Insights and Implications

### Training Stability as a Critical Design Consideration

Our analysis reveals that training stability varies across divergence-similarity combinations, with some exhibiting unpredictable failure modes. Specifically, the pairing of KL with distance-based similarity in supervised contrastive learning appears to lead to training instability-- which could be a possible explanation for why there are more existing methods using a cosine-similarity kernel as shown in the I-Con table as KL is the base distance kernel used in loss formulations.

### Empirical Patterns in Divergence-Similarity Combinations

Our results reveal that different divergence measures exhibit varying effectiveness depending on the chosen similarity measure. The performance differences across these combinations, coupled with their distinct training dynamics, suggest that the choice of divergence cannot be made independently of the similarity measure used in the loss function.

### Total Variation's Strong Performance with Distance-based Measures

We observe that Total Variation distance performs particularly well when combined with distance-based similarity measures across multiple tasks. Beyond its strong final performance, Total Variation demonstrates superior optimization stability compared to KL divergence. This may be related to its $L_1$-norm properties, which provide a more balanced treatment of probability differences and more well-behaved gradients during training.

### Practical Implications

The discovery of training instability in certain divergence combinations has immediate practical implications. Methods that appear competitive in preliminary experiments may fail during extended training or deployment. Our systematic exploration provides practitioners with guidance on which combinations to avoid and which to prioritize for stable, reliable performance.

## Limitations and Future Directions

While our framework successfully unifies diverse representation learning objectives, several limitations warrant discussion. 

Firstly, we did not thoroughly test the Wasserstein distance primarily because of computational overhead. Future work may continue along this direction because of theoretical reasons for why Wasserstein is superior to KL, such as was analyzed by the WGAN paper.

The theoretical understanding of why certain divergences exhibit training instability in specific formulations remains incomplete. A deeper analysis of the optimization landscape, gradient dynamics, and convergence properties under different divergence choices would strengthen the framework and potentially suggest modifications to improve stability.

Finally, while we created one new loss for Supervised Contrastive Learning in particular by swapping out the cosine similarity kernel for the euclidean kernel, this can be done for any loss which exists which is missing one of the two kernels, such as SimCLR. 

## Conclusion

The Beyond I-Con framework challenges the field's implicit assumption that KL divergence is the natural choice for representation learning. By demonstrating that geometry-aware divergences can significantly outperform KL-based methods—and revealing optimization challenges with certain combinations—we open new avenues for developing more principled approaches that explicitly account for both the underlying data manifold structure and training dynamics.

Our systematic exploration of divergence-similarity measure combinations provides a useful framework for analyzing existing methods and discovering new ones. This approach helps researchers understand how these design choices interact and may guide the selection of appropriate combinations for specific applications, with particular attention to optimization stability.

The future of representation learning lies not in finding the single "best" divergence, but in developing principled frameworks for selecting divergences that align with the geometric properties of the data, the specific requirements of the downstream task, and the practical constraints of reliable optimization. The Beyond I-Con framework provides the theoretical foundation for this endeavor, while our empirical results—including the discovery of training instability patterns—demonstrate its practical importance.

As the field continues to grapple with more complex data modalities and geometric structures, we anticipate that divergence-aware approaches will become increasingly important. Our framework provides both the theoretical tools and empirical evidence needed to guide this evolution toward more geometry-aware and optimization-robust representation learning methods.