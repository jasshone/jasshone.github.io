# Composing Diffusion Models

Hello! Just reading some more diffusion models papers, and wanted to write brief summaries for each of them as an aid for learning.

## Compositional Visual Generation with Composable Diffusion Models - Liu et al. https://arxiv.org/pdf/2206.01714

- The goal is to increase performance on complex images by using different diffusion models to capture different subsets of a specification
- They are then composed together to generate an image
- This method allows for significantly more complex combinations than in training
- We know the basic rundown of diffusion models so I will not repeat it here.
- EBMs are a class of generative models where the data distribution is modeled with a probability density.

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/ec8f356b-f180-41da-b6dc-8e06fcff1054">

Where the exponent $E_{\theta}(x)$ is a learnable neural network. 

- The sampling procedure is functionally the same as that for diffusion models, as you can see in the following equation:

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/b3940dcd-7d01-4d2e-9a58-90931c3a5353">

- The key innovation is then treating EBMs and diffusion models as the same thing, where the noise prediction network essentially is equivalent to predicting the change in energy.
- A trained diffusion model can be viewed as an "implicitly parameterized EBM".
- EBMs have been shown previously to have good capabilities in compositional generation.
- The composed distribution is as follows:

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/3529cd2d-89a6-4fd2-bda5-906e63a142d5">

- intuitively, we can try to do the same thing for diffusion models as follows:
<img width="278" alt="image" src="https://github.com/user-attachments/assets/3d948c10-f68c-4d4e-8116-d994ce538024">

- Inspired by EBMs, the authors define two compositional operators, conjunction AND and negation NOT to compose diffusion models
- They train a set of diffusion models representing conditional probability distribution x given concept c and an unconditional probability distribution

### AND

<img width="80%" alt="image" src="https://github.com/user-attachments/assets/37bcc254-bbfb-4aab-9114-aa676bca56b6">

- Equation 9 states that the probability that image x is generated given concepts 1-n is proportional to the probability that x and all of the concepts are generated, which is the product of the probability of x * the probability of the concepts given x (given the concepts are conditionally independent of each other given x)
- With bayes rule, we can substitute $p(c_i|x)$ with $\frac{p(x|c_i)}{p(x)}$ (because the equation states that the proabbility is proportional to rather than equal to) and get equation 10
- To generate the noise prediction, we adding the noise given x and timestep t, and for each concept, adding the noise predicted given the concept and subtracting the original noise, and taking a weighted sum over all concepts (the weight is called temperature scaling)
- To compose an image with this updated noise, we do the same thing as if we had noise from only one diffusion model (equation 12)

### NOT

- the negation of a concept can be ill-defined, such as the negation of "dark" being "bright" or random noises
- It's necessary to have conditioning on other concepts as a result.
- As such, the authors refactorize the joint probability distribution as

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/6502ba9d-2eab-47dd-a739-48aa9450f991">

<img width="90%" alt="image" src="https://github.com/user-attachments/assets/dfa92408-1958-457d-bdd2-a948d302845f">

### Experiments

- Used CLEVR, Relational GLEVR, FFHQ datasets which are objects, relational description images, and real-world human faces respectively
- Evaluated using Binary classification, on three different settings: generating with 1 component, or conditioning on single concept, generating with 2 components, or generating with 3 components
- Also evaluated using FID
- Compared with StyleGAN2-ADA, LACE, GLIDE, EBM
- Composed results generally higher quality images with correct object relations
- Three failure cases: (1) pre-trained diffusion models do not understand certain concepts, (2) diffusion models confuse attributes of objects, (3) the composition does not work, which often happens when objects are in the center of the images

### Conclusion

- Limitation of approach: composing multiple models only works if they are instances of the same model
- Limited success when composing diffusion models trained on different daasets
- EBMS can succesfully compose multiple separately trained model

## Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC - Du et al. 2023 https://energy-based-model.github.io/reduce-reuse-recycle/

- the goal is to repurpose diffusion models without finetuning to a variety of downstream tasks.
- Like in the previous paper, an intuition is to borrow ideas from composing EBMs.
- However, the formula given previously for how the EBM parametrizes a distribution is slightly incorrect, and the correct formulation is:

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/61cbd47b-1183-498c-ab3e-1339d558e40b">

where $Z(\theta)$ is

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/a09999f9-7280-42b4-83e7-0fe038da6470">

or the area under the curve of the probability distribution (so that the probability adds to 1)

- By not modelling the normalization constant, we can no longer efficiently compute likelihoods or draw samples
- this complicates training because most generative models are trained by maximizing likelihood
- To try to get likelihoods or sample from the EBM, we must rely on approximate methods such as MCMC.
- MCMC with Unadjusted Langevin Dynamics is basically the same thing functionally as diffusion models, as previously noted.
- The rewritten training objective of diffusion models is as follows
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/4cd561f4-fc03-4cc5-be2a-d0c346c3a39e">

### Controllable Generation

- Learning a conditional diffusion model is learning $p_{theta}(x|y;t)$.
- Exploiting Bayes rule leads us to find for $\lambda = 1$: <img width="80%" alt="image" src="https://github.com/user-attachments/assets/db7b085f-881c-449f-9025-1a06e4945bbe">
- The intuition of this method is to essentially learn a classifier for different y which can guide the generations of x given y
- in practice, it is beneficial to make $\lambda >1$.

- There is another way to do this which is to not learn this explicit model but an implicit model of x|y
- The equation ends up being:

<img width="80%" alt="image" src="https://github.com/user-attachments/assets/0335a2bf-4275-4708-8053-de3076004ab9">

- The first method allows you to train a bunch of different classifiers and attach them to the generative model
- The second method has better performance but it's harder and more expensive to do.


