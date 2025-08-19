# Some Thesis Defenses I went to 

I haven't posted as much with research ramping up and other things as well, but I wanted to write down some of my thoughts for some really cool thesis defenses I went to today and a while ago from my lab. I may add some more if I go to any more.

## Defense 1

The first researcher focuses on combining classical robotics with intelligent planners, and worked jointly with the meche dept and my lab. The core idea of her research is how to plan under uncertainty, and the principle that "if the model doesn't fail irreparably then the situation is salvagable/it can continue until it reaches the goal".

Classical planners work very well in situations where the world model is complete and doesn't have a lot of flaws. This can be achieved through performing search on the space of possible actions within some constraints such as collisions. The researcher detailed her work on developing a specific kind of search which basically finds "cones" within action space of possible actions which obey sets of constraints and move the robot towards the goal, and then searches the cones resulting from the potential action space after performing the actions from the previous cone, and so on, until it reaches the goal state.

However, under situations with uncertainty where the world model could be imperfect such as in cases of occlusion or unknown objects, these classical planners may not perform very well. Her idea in these situations is to designate "danger zones" for actions which are a certain distance in action space away from making an irreversible move which causes sure failure, such as breaking a glass vase.

## Defense 2

The second researcher is very much in a different area of approaching robotics, specifically focusing on the capabilities of EBMs and diffusion models. I read through some of their papers in an earlier post so it was interesting seeing them summarize all of their work.

Energy based models try to model the entire landscape of objects and how well they fit within some descriptor or set of objects. Diffusion models can be thought of as very closely related to EBMs, except that they try to predict the denoising step rather than the actual energy landscape.

This researcher focused on the fact that EBMs and diffusion models are very tightly related to allow for new ways to compose these models, and as I previously discussed in the other blog post, doing this kind of composition allowed much better results when conditioning on multiple distributions, such as "pink flower" and "orange mountain" jointly.

Another branch of their research was applying diffusion models to RL and imitation learning. They modelled this as a rectangle along the diagonal of a box -- theoretically, because robot demonstration data is much less representative of the space of possible robot actions, training subunits of diffusion models to model each axis allowed for composing these models to cover a much larger distribution of data.

The final sector of their research was focusing on composing different large models together in order to get good results on out-of-distribution data. One interesting example they gave was jointly using and LLM and CLIP to refine a description of an image, where the CLIP feedback was essentially used to perform gradient descent on the description until it matched the image. Another cool example was using video generation models and VLMs jointly to solve a task from a task description in few shot or zero shot. 

## Conclusions

I greatly enjoyed attending both defenses and I feel like I learned a lot. I just got the advice to read 10-20 papers a week in a conversation with another researcher in my lab, and I think attending these defenses has made me more motivated to do so. I also hope that I have the chance to do more implementations of various architectures on my own as the school year starts back up again.
Thanks for reading, and see you in the next blog!
