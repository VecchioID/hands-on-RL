## A2C
Advantage Policy Gradient, an paper in 2017 pointed out that the difference in performance between A2C and A3C is not obvious.

The Asynchronous Advantage Actor Critic method (A3C) has been very influential since the paper was published. The algorithm combines a few key ideas:

- An updating scheme that operates on fixed-length segments of experience (say, 20 timesteps) and uses these segments to compute estimators of the returns and advantage function.
- Architectures that share layers between the policy and value function.
- Asynchronous updates.

## A3C
Original paper: https://arxiv.org/abs/1602.01783