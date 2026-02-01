# Week 01: Robot Room Search Using Markov Decision Processes

# 1. Problem Description and Motivation
This project models a robot searching rooms in a building to locate a target while minimizing expected total search cost.
The robot does not know the target’s location and instead maintains a probability-based belief over all rooms, which is updated after each search.

The problem is formulated as a Markov Decision Process (MDP) and solved using policy iteration.

# Problem Description

1) The robot chooses one room to search at each step

2) Each room has a fixed search cost

3) Searching a room either:

    a) finds the target (task ends), or

    b) fails and updates the belief

4) The objective is to minimize the total expected cost until the target is found

# MDP Formulation
State Space
The state is the robot’s belief about the target location:
$$
s = (p_1, p_2, \dots, p_K), \quad \sum_{i=1}^{K} p_i = 1
$$
where p_iis the probability that the target is in room i.
The belief space is discretized using step size 0.1 to obtain a finite MDP.

# Action Space
$$
A={Search Room 1,Search Room 2,…,Search Room K}
$$

# Transition Model

If room 𝑖 is searched:

The target is found with probability 
$$
𝑝
𝑖
p
i
	​

 (terminal state)
 $$

Otherwise, the belief is updated using Bayesian conditioning:

$$
𝑝
𝑖
′
=
0
,
𝑝
𝑗
′
=
𝑝
𝑗
1
−
𝑝
𝑖
,
∀
𝑗
≠
𝑖
p
i
′
	​

=0,p
j
′
	​

=
1−p
i
	​

p
j
	​

	​

,∀j

=i
$$

# Reward Function

Each action incurs a search cost:
$$
R(s, a = i) = -c_i
$$

# Objective

The goal is to find an optimal policy 𝜋∗ that minimizes the expected total search cost incurred until the target is found:
$$
\pi^* = \arg\min_{\pi}
\mathbb{E}\left[
\sum_{t=0}^{T} c(a_t)
\right]
$$

# Equivalent Reward Formulation

The same objective can be expressed in reward-maximization form by defining rewards as negative search costs:

$$
\pi^* = \arg\max_{\pi}
\mathbb{E}\left[
\sum_{t=0}^{T} \gamma^t
R(s_t, \pi(s_t))
\right]
$$

# Discount Factor

$$
\gamma = 1
$$

The discount factor is set to 1 because the robot search task is finite and episodic. The process terminates once the target is found, and there is no need to discount future costs since all costs represent real search effort before termination.

# Solution Method
The MDP is solved using policy iteration, which alternates between:
# Policy Evaluation

$$
V^\pi(s) =
R(s, \pi(s)) +
\gamma \sum_{s'} P(s' \mid s, \pi(s)) V^\pi(s')
$$

This equation is solved iteratively to compute the expected total cost (or reward) when following policy 
𝜋.

# Policy Improvement

Using the evaluated value function, the policy is updated greedily:
$$
\pi_{\text{new}}(s) =
\arg\max_{a \in A}
\left[
R(s,a) +
\gamma \sum_{s'} P(s' \mid s,a) V(s')
\right]
$$
Each state selects the action that maximizes the expected value.

Convergence

Policy evaluation and policy improvement are repeated until the policy no longer changes.

Policy iteration converged in 2 iterations, indicating fast and stable convergence for this finite belief state MDP.

# Results

# Optimal first room to search: Storage

# Expected total search cost: 1.0

# Optimal Search Order

1. Storage

2. Kitchen

3. Bathroom

4. Living Room

5. Office

6. Bedroom

The policy favors low-cost rooms first, even when their probability is lower.

# Key Takeaway

This project shows how MDPs can optimally solve search problems under uncertainty.
The learned policy balances search cost and probability, rather than greedily selecting the most likely room, resulting in a cost-efficient search strategy.

   
