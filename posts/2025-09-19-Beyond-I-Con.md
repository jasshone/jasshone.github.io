# Beyond I-Con: Exploring New Dimensions of Distance Measures in Representation Learning

*Jasmine Shone; project in collaboration with Shaden Alshammari, Mark Hamilton, Bill Freeman*  
*MIT Computer Science*

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

*Results shown as Linear Probing / KNN accuracy percentages*

Notably, while KL divergence performs well with angular similarity measures, Total Variation achieves the highest performance when combined with distance-based measures. This suggests that certain divergence-similarity measure combinations may be particularly effective for supervised contrastive learning.

## Key Insights and Implications

### Empirical Patterns in Divergence-Similarity Combinations

Our results reveal that different divergence measures exhibit varying effectiveness depending on the chosen similarity measure. The performance differences across these combinations suggest that the choice of divergence cannot be made independently of the similarity measure used in the loss function.

### Total Variation's Strong Performance with Distance-based Measures

We observe that Total Variation distance performs particularly well when combined with distance-based similarity measures across multiple tasks. This may be related to its $L_1$-norm properties, which provide a more balanced treatment of probability differences compared to KL divergence, which can be dominated by probability mass on rare events.

### Practical Implications

Some of the newly explored divergence-similarity combinations demonstrate competitive performance with existing methods. This empirical finding suggests that practitioners may benefit from considering these alternative formulations, particularly when working with specific types of data or computational constraints.


## Limitations and Future Directions

While our framework successfully unifies diverse representation learning objectives, several limitations warrant discussion. Our empirical evaluation focuses primarily on computer vision tasks—the generalizability to natural language processing or graph representation learning remains an open question.

The computational overhead of certain divergences may limit their practical applicability in large-scale settings. Future work should investigate more efficient approximation schemes for geometry-aware divergences such as Wasserstein distances, or develop novel divergences that balance geometric awareness with computational tractability.

The theoretical understanding of why certain divergences perform better in specific formulations remains incomplete. A deeper analysis of the optimization landscape and convergence properties under different divergence choices would strengthen the framework.

## Conclusion

The Beyond I-Con framework challenges the field's implicit assumption that KL divergence is the natural choice for representation learning. By demonstrating that geometry-aware divergences can significantly outperform KL-based methods, we open new avenues for developing more principled approaches that explicitly account for the underlying data manifold structure.

Our systematic exploration of divergence-similarity measure combinations provides a useful framework for analyzing existing methods and discovering new ones. This approach helps researchers understand how these design choices interact and may guide the selection of appropriate combinations for specific applications.

The future of representation learning lies not in finding the single "best" divergence, but in developing principled frameworks for selecting divergences that align with the geometric properties of the data and the specific requirements of the downstream task. The Beyond I-Con framework provides the theoretical foundation for this endeavor, while our empirical results demonstrate its practical promise.

As the field continues to grapple with more complex data modalities and geometric structures, we anticipate that divergence-aware approaches will become increasingly important. Our framework provides both the theoretical tools and empirical evidence needed to guide this evolution toward more geometry-aware representation learning methods.