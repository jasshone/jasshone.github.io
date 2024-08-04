# Diffusion Policy in Robotic Learning

I wanted to write this blog as a follow up to my previous blog about diffusion models. 
The papers I will be referencing are: 
https://arxiv.org/pdf/2205.09991, https://arxiv.org/pdf/2303.04137v4. I will be summarizing both papers by section, since they build upon each other.

# Paper 1: Planning with Diffusion for Flexible Behavior Synthesis by Janner et al.

This first paper is a predecessor to the second paper. A lot of the second paper is trying to further improve the first paper's ideas.

## Introduction

- The idea of using a learned model for RL and data-driven decision-making is to let the model learn distributions/environment dynamics and then plug the result into a classical trajectory optimization routine which will work well given that you understand the environment well.
- However, this doesn't actually work that well because if the learned model's output is inaccurate, then the trajectory optimizer could learn an incorrect pairing between the learned model output and its effect on the correct trajectory prediction. 
- As such, oftentimes RL methods with trajectory optimization use model-free approaches which minimize the error of what is fed into the trajectory optimizer
- Methods that do rely on learned model outputs use simple trajectory optimization routines which don't actually learn/change over time, so that wrong model outputs in the beginning of training don't have a lasting impact.
- The paper proposes an alternative approach where the model is designed to not output an encoding of the environment but predict the trajectory directly.
- Additionally, the model should be agnostic to the reward function since we want to apply it to tasks where there is no reward.
- The model proposed by the paper, Diffuser, predicts all timesteps of a plan simultaneously. The idea is to allow for long term planning and also not let an error in a particular step compound over time.
- training the diffusion model is as we discussed in the previous blog, or learning to predict the noise in an image.

## Planning with Diffusion

- Other methods have a model learn the environment dynamics and then use a trajectory optimizer on that output; diffusion policy uses the model to learn both the environment dynamics and plan the trajectory.
- Additionally, any step of the process can have conditioning added so that the output is adaptive to environmental changes.

### A Generative Model for Trajectory Planning

- There's a weird thing that arises if trying to autoregressively predict the next timestep using a diffusion model - since the output is conditioned on both the past timesteps and the goal state, there is no clear temporal ordering in the predictions. To solve this problem, the network predicts all timesteps together
- Diffuser has a single temporal convolution to allow for timesteps to influence each other. The idea is that ensuring the local consistency of the trajectory would be enough to ensure global consistency.
- The architecture used is basically the UNet architecture but with the 2D convolutions for images replaced by 1D convolutions by time. The horizon of images is determined by the model architecture and the dimensionality can change if desired during planning.

### Reinforcement Learning as Guided Sampling

- To solve RL problems with Diffuser, reward has to be incorporated somehow.
- The idea is to train a separate network for reward whose gradients is used as conditioning for the gradient
- This means adding the sum of the gradients to the predicted noise.

### Goal-Conditioned RL as Inpainting

- In image diffusion, inpainting is when the model learns to fill in missing parts of the image.
- The claim is that this can be transitioned into goal-conditioned RL because the goal is to predict a trajectory given state and action constraints, and future state and action constraints may be unobserved (like unobserved pixels in an image) and must be filled in by the diffusion model.

## Properties of Diffusion Planners

- Learned long-horizon planning: Diffuser plans over the entire trajectory, which helps with sparse reward settings where rewards are very few and come at the end such that models that predict the next action struggle
- Temporal compositionality: because Diffuser creates globally coherent trajectories by improving local consistency, it can also stitch together subparts of a trajectory to create new trajectories not in the training data.
- Variable-length plans: the prediction length is determined only by how long the noise vector is.
- Task compositionality: Diffuser is independent of reward function, so the model can be guided by different rewards or even combinations of reward functions through lightweight perturbation functions.

## Experimental Evaluation

The goal is to evaluate 1. the ability to plan over long horizons, 2. ability to generalize to new configurations of goals unseen during training, 3. and the ability to recover an effective controller from heterogenous data of varying quality.

### Long Horizon Multi-task Planning

- They evaluate this in the Maze2D environment where a reward of 1 is given only after you finish traversing a trajectory.
- using the inpainting strategy with diffuser achieves higher score than a reference expert policy
- Diffuser also performs well in a multi-task planner setting in Multi2D because it can just be guided by different reward networks and do fine.
  
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

- applying receding-horizon control, which is a concept in control theory in which the planner plans for multiple future timesteps given information about multiple past timesteps
- treating visual observations as conditioning for the diffusion policy, so that each image is processed by the network once which saves compute/enables real-time predictions
- proposing a new transformer-based diffusion network that minimizes the over-smoothing effects of CNN diffusion models and helps with tasks that need a lot of action changes/changing velocity.

## Diffusion Policy Formulation

### A. Denoising Diffusion Probabilistic Models
- Another name for denoising for diffusion models is "Stochasic Langevin Dynamics"
- Like we discussed in the previous blog, the model learns to predict the difference in noise between the image at timestep k and k-1, and in doing so, learns to denoise the image.
- Another way of stating this is as follows:
<img width="40%" alt="image" src="https://github.com/user-attachments/assets/f0d7e8c4-7760-466a-8b15-fd29e03821b6">

where $`\varepsilon_{\theta}(x, k)`$ or the noise prediction network, predicts the gradient field $`\Delta E(x)`$ and $`\gamma`$ is the learning rate.

### B. DDPM Training
- Like we found in the previous blog, the training is basically choosing a timestep, adding random Gaussian noise with variance which is correct for timestep t, and then asking the model to predict the noise.
- The reason why L2 or MSE is used is because minimizing this loss minimizes the variational lower bound of the KL divergence, which is a measure of how different two distributions are. We want to minimize the KL divergence between the actual noise and the predicted noise.

### Diffusion for Visuomotor Policy learning
- The two major modifications made for robot learning is 1. changing the output to represent robot actions 2. making the denoising process conditioned on the input observation $`O`$.
- The authors add closed-loop action-sequence prediction, which means that at time step $t$ the policy takes the last
$`T_O`$ steps of observation data as input and predicts $`T_p`$ steps of actions, of which $`T_a`$ steps of actions are executed without re-planning.
  - In other terms, $`T_O`$ is the observation horizon, $`T_p`$ is the action prediction horizon, and $`T_a`$ is the action execution horizon.
  - The idea behind this is to take into account more of the past and action sequences, predict into the future (to allow for more long-term planning), and allow for changes to be made if necessary (by making a shorter action execution horizon than the prediction horizon)
- The authors use a DDPM to approximate the conditional distribution of actions given observations instead of the joint distribution of actions and observations. This leads the model to be able to predict actions without needing to also predict future observations resulting from the actions, which decreases compute
  - Additionally, conditioning on observations allows for the visual encoder to be trained end-to-end, which will be discussed in more detail later.

Fig 3 illustrates the overall pipeline.

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/72663b5b-fbfc-4787-ad38-6393ec8b7a40">

## Key Design Decisions

### Network Architecture

