# Diffusion Policy in Robotic Learning

I wanted to write this blog as a follow up to my previous blog about diffusion models. 
The original paper is linked here: https://arxiv.org/pdf/2303.04137v4. I will be summarizing the paper by section and noting anything I found particularly interesting.

## Introduction

- Policy learning from demonstration is basically learning how to, from some observations of a scene, convert that into appropriate actions.
- However, policy learning for robotics is different than other supervised learning tasks because there is multimodal data, sequential data, and the need for high precision. These factors combine to make robot learning quite challenging
- Past work tries to solve this by changing how actions are represented to the robot or how the policy is represented.
- Diffusion policy tries to solve this by not directly outputting an action but inferring the `action-score gradient`

This is kinda confusing so far, but I think figure 1 helps with developing an intuition for what this means.

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/d09e7908-5b22-44c4-a7cd-dd6c3d956181">

Explicit Policy, as you can see in subfigure a, entails directly predicting an action. These actions can be represented as a scalar (e.g. joint angle) which is normal regression, a member of a gaussian cluster (grouping the data into different normal clusters and then predicting action depending on which cluster the data belongs to)
, and predicting action categories. In short, input: observation, output: action.

Implicit Policy, as shown in subfigure b, chooses the action by first learning an energy function given an observation and action, and then choosing the action with the minimum energy. In short, input: observation, action, output: energy. Choice of action = argmin of the energy of the actions given the current observation. 

Diffusion Policy, tries to learn, given an initial action of gaussian noise, to convert some of that noise into an action for K total iterations. 
The benefits of this include allowing for the expression of normalizable distributions, high-dimensional output spaces, and stable training.
More details can be found in the paper. 

In addition to introducing diffusion policy to robotics, the authors also state the following contributions:

- applying receding-horizon control, which is a concept in control theory that predicts future costs/disturbances/constraints while generating the current policy
- treating visual observations as conditioning for the diffusion policy, so that each image is processed by the network once which saves compute/enables real-time predictions
- proposing a new transformer-based diffusion network that minimizes the over-smoothing effects of CNN diffusion models and helps with tasks that need a lot of action changes/changing velocity.

## Diffusion Policy Formulation

### A. Denoising Diffusion Probabilistic Models
- Another name for denoising for diffusion models is "Stochasic Langevin Dynamics"
- Like we discussed in the previous blog, the model learns to predict the difference in noise between the image at timestep k and k-1, and in doing so, learns to denoise the image.
- Another way of stating this is as follows:
<img width="187" alt="image" src="https://github.com/user-attachments/assets/f0d7e8c4-7760-466a-8b15-fd29e03821b6">
where $`\Epsilon_{\theta}(x, k)`$ or the noise prediction network, predicts the gradient field $`\Delta E(x)`$ and $`\gamma`$ is the learning rate.

### B. DDPM Training
- Like we found in the previous blog, the training is basically choosing a timestep, adding random Gaussian noise with variance which is correct for timestep t, and then asking the model to predict the noise.
- The reason why L2 or MSE is used is because minimizing this loss minimizes the variational lower bound of the KL divergence, which is a measure of how different two distributions are. We want to minimize the KL divergence between the actual noise and the predicted noise.

### Diffusion for Visuomotor Policy learning
- The two major modifications made for robot learning is 1. changing the output to represent robot actions 2. making the denoising process conditioned on the input observation `O`.
- The authors add closed-loop action-sequence prediction, which means that at time step $t$ the policy takes the last
$`T_O`$ steps of observation data as input and predicts $`T_p`$ steps of actions, of which $`T_a`$ steps of actions are executed without re-planning.
- In other terms, $`T_O`$ is the observation horizon, $`T_p`$ is the action prediction horizon, and $`T_a`$ is the action execution horizon.
- The idea behind this is to take into account more of the past and action sequences, predict into the future (to allow for more long-term planning), and allow for changes to be made if necessary (by making a shorter action execution horizon than the prediction horizon)

