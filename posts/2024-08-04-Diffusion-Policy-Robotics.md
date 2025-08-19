# Diffusion Policy in Robotic Learning

I wanted to write this blog as a follow up to my previous blog about diffusion models. 
The papers I will be referencing are: 
https://arxiv.org/pdf/2205.09991, https://arxiv.org/pdf/2303.04137v4. I will be summarizing both papers by section, since they build upon each other.

## Paper 1: Planning with Diffusion for Flexible Behavior Synthesis by Janner et al.

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
- The idea of this method of adding reward is to train a separate network for reward whose gradients is used as conditioning for the gradient
- This means adding the sum of the gradients to the predicted noise.

### Goal-Conditioned RL as Inpainting

- In image diffusion, inpainting is when the model learns to fill in missing parts of the image.
- The claim is that this can be transitioned into goal-conditioned RL because the goal is to predict a trajectory given state and action constraints, and future state and action constraints may be unobserved (like unobserved pixels in an image) and must be filled in by the diffusion model.
- The idea of this method is to condition on future state and action constraints

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
- In contrast, even the best model-free algorithm drops significantly in performance when a single-task model is applied to a multi-task dataset

### Test-time flexibility

- They create three block stacking tasks: (1) Unconditional stacking, which is building a block tower that is as tall as possible (2) Conditional stacking, which is to construct a block tower with a specified order of blocks, and (3) rearrangement, which is to match a set of reference blocks' locations in a novel arrangement.
- The first task has no conditioning/perturbation function, the second and third condition on maximizing the likelihood of the trajectory's final state matching the goal configuration, and a contact constraint between the end effector and a cubeduring stacking motions.
- Diffuser outperforms prior methods

### Offline RL

- Both inpainting and sampling are used
- performance is better than other general-purpose RL techniques but worse than the best offline techniques for single-task performance
- trying Diffuser + a conventional trajectory optimizer found that the combination performed no better than random, so the effectiveness of Diffuser is from the coupled modeling and planning.

### Warm-Starting Diffusion for Faster Planning

- The intuition is to use past plans for new plans at later timesteps to avoid computational cost
- this can be done by running a limited number of forward diffusion steps and then a corresponding number of denoising steps to get an updated plan.


## Paper 2: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

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

where $\varepsilon_{\theta}(x, k)$ or the noise prediction network, predicts the gradient field $\Delta E(x)$ and $\gamma$ is the learning rate.

### B. DDPM Training
- Like we found in the previous blog, the training is basically choosing a timestep, adding random Gaussian noise with variance which is correct for timestep t, and then asking the model to predict the noise.
- The reason why L2 or MSE is used is because minimizing this loss minimizes the variational lower bound of the KL divergence, which is a measure of how different two distributions are. We want to minimize the KL divergence between the actual noise and the predicted noise.

### Diffusion for Visuomotor Policy learning
- The two major modifications made for robot learning is 1. changing the output to represent robot actions 2. making the denoising process conditioned on the input observation $`O`$.
- The authors add closed-loop action-sequence prediction, which means that at time step $t$ the policy takes the last
  $T_O$ steps of observation data as input and predicts $T_p$ steps of actions, of which $T_a$ steps of actions are executed without re-planning.
  - In other terms, $T_O$ is the observation horizon, $T_p$ is the action prediction horizon, and $T_a$ is the action execution horizon.
  - The idea behind this is to take into account more of the past and action sequences, predict into the future (to allow for more long-term planning), and allow for changes to be made if necessary (by making a shorter action execution horizon than the prediction horizon)
- The authors use a DDPM to approximate the conditional distribution of actions given observations instead of the joint distribution of actions and observations. This leads the model to be able to predict actions without needing to also predict future observations resulting from the actions, which decreases compute
  - Additionally, conditioning on observations allows for the visual encoder to be trained end-to-end, which will be discussed in more detail later.

Fig 3 illustrates the overall pipeline.

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/72663b5b-fbfc-4787-ad38-6393ec8b7a40">

## Key Design Decisions

### Network Architecture

- They try the 1D temporal CNN from the first paper with a few modifications:
  - They only model the conditional distribution with Feature-wise Linear Modulation (another influential paper, may write a blog about this later) and the iteration of denoising $k$.
  - They only predict the action trajectory instead of the observation-action trajectory
  - They removed inpainting-based goal state conditioning due to incompatibility with the receding prediction horizon framework.
  - It worked well on most tasks but it performed poorly when the desired action sequence changes quickly and sharply through time such as velocity command

- They then try a new transformer architecture in which actions with noise are passed as input tokens for transformer decoder blocks, with the sinusoidal embedding for the iteration k of diffusion prepended as the first token.
  - The observation is transformed using an MLP to an observation embedding sequence, and then passed to the transformer decoder stack
  - The gradient for each timestep is predicted by each corresponding output token of the decoder stack
  - The performance was better than the CNN but more sensitive to hyperparameters
 
  ### Visual Encoder
  
  - visual encoder maps raw image sequence to a latent embedding and is trained end-to-end with the diffusion policy. Different camera view use separate encoders and images in each timestep are encoded independently.
  - ResNet-18 architecture was used as the encoder, with the modifications of replacing the global average pooling with a spatial softmax pooling to maintain spatial info, and replacing BatchNorm with GroupNorm.
 
  ### Noise Schedule + Inference
  
  - Noise schedule is important for better representing action signal change frequency, and they find that Square Cosine Schedule gives them the best performance
  - They use the same trick as the original diffusion paper to denoise multiple steps at a time for faster inference
 
  ## Intriguing Properties of Diffusion Policy

  - Modelling multi-modal action distributions, through having sampling/randomness enable multiple correct solutions to be represented
  - Synergy with Position Control, or that it works better with predicting positions than velocity.
  - Benefits of action-sequence prediction: or the diffusion model being able to express variable length sequences well without compromising expressiveness of the model.
    - if there are multimodal distributions for the correct action, other models with only one timestep of prediction could switch between different solutions.
    - Idle actions sometimes occur where demonstrations are paused. Models with only predictions in a single timestep forward can overfit to this pausing behavior.
  - Diffusion models are more stable to train than energy-based intrinsic models.

  ## Evaluation

  - Diffusion Policy is evaluated on a suite of robot learning tasks which include simulation tasks (e.g. BlockPush) and real world tasks (e.g. Push-T)
  - Key findings:
    - Diffusion Policy can express short-horizon multimodality (multiple ways of achieving the same immediate goal) and does it better than other methods
    - Diffusion Policy can express long-horizon multimodality, or the completion of different sub-goals in different orders, better than other models
    - Diffusion Policy can better leverage position control than velocity control. The baselines they compare with work best with velocity control, and so does most of literature.
    - A longer action horizon helps policy predict consistent actions and compensate for idle portions of the demonstration, but too long a horizon reduces performance due to slow reaction time.
    - Robustness against latency: receding horizon position control helps reduce latency gap by image processing, policy inference, and network delay.
    - stable to train: optimal hyperparameters are mostly consistent across tasks, vs energy-based intrinsic models.
   
  ### Real World Eval
  - They found that the CNN variant with end-to-end vision encoders performed better
  - In the Push-T task, where the model must push a "T" object into a specific configuration, they found that diffusion policy is robust to perturbations, such as waving a hand in front of the camera, shifting the object, and moving the object after the task was complete but the robot was still moving to the end-zone. This indicates that diffusion policy may be able to synthesize novel behavior in response to unseen observations.
  - In the Mug Flipping task, they found that diffusion policy is better able to handle complex 3D rotations which are highly multimodal compared to an LSTM-GMM policy
  - In the Sauce Pouring and Spreading task, they found that diffusion policy is better able to work with non-rigid objects, high-dimensional action spaces, and periodic actions (with idle periods).


# Conclusions and Thoughts

These were very information dense papers and had a lot of high quality observations and analysis. I will likely reread them at some point and try to understand them even further; this was a really great read and I learned a lot. Thanks for reading, and see you in the next blog!
