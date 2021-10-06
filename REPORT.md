## Project 3: Collaboration and Competition

</hr>

### Environment Details
This is an environment provided by Udacity that is similar, but not identical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-Agents GitHub page.

</br>

![tennis](https://github.com/p-Cyan/Udacity_DeepRL_P3_Collaboration_and_Competition/blob/main/images/Agent%20demonstration.gif)
</br>
</br>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.


### Termination Condition
The environment is considered solved when the average (over 100 episodes) of those **scores** is at least +0.5.

## Learning Algorithm

This environment was solved using a DDPG algorithm with multi agent implementation. While udacity suggested MADDPG with [this](https://arxiv.org/pdf/1706.02275v4.pdf) implementation, the basic model used here was proved more than sufficient to solve the environment.

### Deep Deterministic Policy Gradients (DDPG) algorithm.

The algorithm used here is a Deep Deterministic Policy Gradient (DDPG). A DDPG is composed of two networks : one actor and one critic. During a step, the actor is used to estimate the best action, ie argmaxaQ (s, a); the critic then use this value as in a DDQN to evaluate the optimal action value function.

Both of the actor and the critic are composed of two networks. On local network and one target network. This is for computation reason : during backpropagation if the same model was used to compute the target value and the prediction, it would lead to computational difficulty.

During the training, the actor is updated by applying the chain rule to the expected return from the start distribution. The critic is updated as in Q-learning, ie it compares the expected return of the current state to the sum of the reward of the choosen action + the expected return of the next state.

### Code implementation
The code for ddpg agent used here is derived from the ddpg-pendulum project made by Udacity. Since the model has to perform using multiple agents, it has been slightly adjusted for being used with the this environment. ( editted the code such that both agents learn and update simultaneously, and added slight noise among agents to ensure learning occurs).
We want a single network to be able to play as either side, so both of the agents utilize same actor and critic networks to use and update. 

The code consists of:
- model.py : This python file contains the model framework defined using pytorch.
- ddpg_multiple_agents.py : This python file contains the implementation of ddpg agent.
- Continuous_Control.ipynb : This is the jupyter implementation of the file that utilizes the ddpg agent to solve the environment.

The saved model weights are:
- checkpoint_actor.pth : Weights of the trained actor model
- checkpoint_critic.pth : Weights of the trained critic model

### Model architecture

Due to limitations in hardware and GPU hours in udacity workshop, only small networks have been trained. Luckily, these networks seemed to be sufficient to learn the tasks.

The working version of models are as follows:

```
Actor_Network(
  (fc1): Linear(in_features=33, out_features=128 )
  (fc2): Linear(in_features=128, out_features=128 )
  (out): Linear(in_features=128, out_features=4 )
)
```
```
Critic_Network(
  (fc1): Linear(in_features=33, out_features=128 )
  (fc2): Linear(in_features=128, out_features=128 )
  (out): Linear(in_features=128, out_features=1 )
)
```

### Hyper parameters

The training hyperparameters are as follow :
- Buffer size : 100,000
- Batch size : 128
- GAMMA : 0.99
- TAU : 0.001
- learning rate actor : 0.001
- learning rate critic : 0.001
- weight decay : 0

### Results

The model starts off slow, but it speeds up very fast the moment it starts learning. In this case it took 938 episodes to solve the environnment.

![results](https://github.com/p-Cyan/Udacity_DeepRL_P3_Collaboration_and_Competition/blob/main/images/History.JPG)

![plot](https://github.com/p-Cyan/Udacity_DeepRL_P3_Collaboration_and_Competition/blob/main/images/graph.JPG)

## Future
The performance of the agents might be improved by considering the following:

- Adding prioritized replays
- Hyperparameter optimisation 
- Alternative learning algorithms: 
  - PPO
  - [MADDPG](https://arxiv.org/pdf/1706.02275v4.pdf)
