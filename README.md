## Udacity Deep Reinforcement Nanodegree
</hr>
This repository contains my project submission for project 3 of Udacity's [Deep RL Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

## Project 3: Collaboration and Competition

</hr>

### Environment Details
This is an environment provided by Udacity that is similar, but not identical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-Agents GitHub page.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.


### Termination Condition
The environment is considered solved when the average (over 100 episodes) of those **scores** is at least +0.5.

The method used is an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm. A multi agent implementation of DDPG has been used (same implementation used in P2: Continuous control but with slight modifications) , where both agents update the same actor and critic network.
The idea is that the same agent should be able to solve the environment irrespective of what player it is.

## Getting Started

</hr>

### Installation requirements
- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).</br>
- You then have to clone this project and have it accessible in your Python environment</br>
 - Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)
- Then, Follow the instructions in tennis.ipynb to get started with training your own agent!

### Misc : Configuration used
This agent has been trained on an Acer Predator Helios 300 PC with an i7 8th generation processor, 16 GB available RAM, GTX 1060 2 GB graphic card.
