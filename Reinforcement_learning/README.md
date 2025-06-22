# Reinforcement Learning Basics

Policy - Defines the behaviour of the agent and it's interaction with the environment. 

Reward Signal - Maximizes the short term success 

Value Function - Maximizes the long term success

Model - Sometimes environment can be complex so we use some mathematical models to simulate the agentic system. 


# Multi-arm bandit problem 

It is toy example of compex reinforcement learning problems. It is modelled as multi-lever game where pulling one particular lever increases the probability of winning, however we need to find that lever and we don't have initial states. 

Value action method - In this method, the agent interacts with environment, we reward it on the basis of correct action. We see the long term behaviour of the value action for each lever and that's how we get the expected probability to choose the best lever. 

qi(a) = 1 if you win
qi(a) = 0 if you lose

This can also be linked to average law over large numbers since we tend to get the estimation of most probable winner with large number of interactions. Note - we do not use greedy search i.e. we do not always choose the most probable rewarding lever during interactions since it can leave us with a local minima. 
We use exploration and exploitation. 
At the end, 

qi(a) = (average over t)qit(at)

we get all the average value action for each lever i.e. the probability of most rewarding lever. 


# Markov Decision process

 - Agent-Environment Interface -
 State - Tells about the state of the system like temperature, energy level, etc. 
 Action - Immediate task/action that agent should do 
 Policy - Action that needs to be done by the system depending on the state. /pi(a|s)

- Rewards and Expected Returns
Rewards - Immediate rewards (rt) given to the agent depending on the action  taken at time step t
Expected returns - Average of all the rewards from time step t to end time step T (gt = rt + r_{t+1} + .. + r_T)

This gives rise to model problems in two ways 

- Episodic - In this, you know the time step at which agent stops interacting  like chess. 
- Continuous - We don't know the time at which agent stops interacting such as in exploration task where you continuosly try to find information. Therefore, in continuous system, we try to decrease the importance of reward from future in the expected return by introducing an exponetial decreasing term \gamma as a factor. 
G_t = /sum_{k = 0}{/infinity}\gammaˆkR_{t+k}


- Markov Property
This says rewards just depend on the previous state i.e. the future depends on the present and not the past.  
P(S_{t+1}, R_t| S_t, A_t) =  P(S_{t+1}, R_t| S_0, A_0, S_1, A_1, .. , S_t, A_t)

- Markov Decision Processes
Any reinforcement learning task which has markov property, is called markov decision process. 


# Value Functions

State value function - It tells about the expected return at the state to the future states. 

v_{/pi} = E_{\pi}(G(t) | S) = E_{\pi}(\sum_{k = 0}{\infinity}\gamma_{k}R_{t+k}| S)
Action value function - It tells about the expected return at the state to the future states, given some action taken. 
a_{/pi} = E_{\pi}(G(t) | S, A) = E_{\pi}(\sum_{k = 0}{\infinity}\gamma_{k}R_{t+k}| S, A)


Bellman equation for optimal value function 

v_{\pi}(s) = \sum_{A}P(\pi(A|S) \sum_{s', r} P(S', r|S, A) (r + \gamma v_{\pi}(s')))

Optimal policy (or optimal action value function)

v_{*}(s) = max_{A} \sum_{s', r} P(S', r|S, A) (r + \gamma v_{*}(s'))


# Bellman derivation 

![Derivation](Bellman_derivation.png)



# Dynamic Programming 


## Prediction problem 
- Policy Evaluation - For a given policy and state, policy evaluation is written as v_{\pi}(s) which tells about values returned by that policy. 

v_{\pi}(s) = \sum_a \pi(a|s) \sum_{s', r} p(S', R|S, A)(r + \gamma*v_{\pi}(S'))

To find the state value function, we usually start with zero, and iterate to infinity to find the correct state values. 

v_{k+1} = \sum_{a} \pi(a|s) \sum_{S', R} p(S', R|S, A) (R + \gamma v_{k}(s'))


So called Iterative policy evaluation 


## Control Problem
- Policy optimization 

\pi = argmax_{a} \sum_{S', R} p(S', R| S, A) (R + \gamma v_{k}(S'))


v_{*} (s) = max_{a \epsilon A} q_{\pi^{*}}(s, a)
= max_{a \epsilon A} \sum_{S', R} p(S', R|S, A) (R + \gamma v_{*}(s'))


v_{k+1} (s) = max_{a \epsilon A}  \sum_{S', R} p(S', R|S, A) (R + \gamma v_{k}(s'))

This method is called as value iteration. It reduces the 2 step of policy evaluation and policy iteration into one single step. 

Policy iteration 
\pi_{0} -> v_{\pi_{0}}
v_{\pi_{0}} -> \pi_{1}

Dynamic programming requires the prior knowledge of the environment and it's interaction with the environment i.e. transition probabilities. 



# Monte Carlo methods 

In dynamic programming, we need to know all the transition probabilities i.e. know about the environment but this is not ideal in real life cases. Therefore, we require a mathematical tool that helps to identify the optimal policy while interacting with environment and not knowing probabilities of agent interacting with environment. In Monte Carlo, agent continuously interacts with environment and updata it's action value function. After several trials, we reach an optimal action value function and hence optimal policy. 

\pi_{0} -> q_{\pi_{0}}(S, A)
q_{\pi_{0}}(S, A) -> \pi_{1}

However, in this case, there might be a case our agent might be stuck with some kind of local mazima. Therefore, we use epsilon greedy policy which allows exploration and not just exploitation. 

There are two kinds of Monte Carlo methods - 
On-policy - The policy with which agent interacts to generate data is same as the target policy. 

Off-policy - The policy with which agent interacts to generate data, known as behaviour policy is different from target policy. (Need to understand more!!!)

Interactive Example to demonstrate on-policy Monte-Carlo control: [link](https://claude.ai/public/artifacts/1e531122-1b45-4337-86a2-df8bc2ac531b)

Interactive Example to demonstrate Importance Sampling Ratio used in off-policy Monte-Carlo methods (not the focus of this lecture): [link](https://claude.ai/public/artifacts/8a773548-d360-418b-a20b-d427dc7a1952)


# Temporal difference learning

It is a combination of dynamic programming and monte carlo methods, which gives the benefit of not knowing the interaction with environment and not need to wait for the whole episode to complete. 

## Prediction problem 

### Monte Carlo method 

From Monte Carlo method, we know how to calculate the state value function 

We calculate state value function 
V_{k+1}(S) =\frac{G_0 + G_1 + G_2 + ... + G_{k-1} + G_{k}}{k+1}

V_{k+1}(S) = V_k(S) + \alpha*(G_k - V_k(S))


V(S_{t+1}) = V(S_t) + \alpha*(G_t - V(S_t))

V(S_t) = V(S_t) + \alpha*((R_{t-1} + R_t + R_{t+1} + ...) - V(S_t))

However, we can't wait for the episode to end. 


### Dynamic programming
Therefore, we used the concepts from dynamic programming (bootstrapping) to estimate the state value function as a function of immediate state value function

G_t = \gamma R_{t+1} + \gamma( R_{t+2} + ...)
G_t = R_{t+1} + \gamma (V(S_{t+1}))


So now the prediction problem looks like 
V(S_{t+1}) = V(S_t) + \alpha*(( R_{t+1} + \gamma (V(S_{t+1}))) - V(S_t))

Now, we don't need to wait for the whole episode to be over. 


## Control Problem

If we want to prediction the value of state-action pair, we calculate state action value Q(S, A). 

### On-Policy Control

If the agent applies the policy with which it has learnt, it's called on-policy. For example, a person rides a bicycle and improves it's skill while driving. 

\textbf(SARSA Algorithm)
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (( R_{t+1} + \gamma (Q(S_{t+1}, A_{t+1}))) - Q(S_t, A_t))


### Off-policy control 
If the agent applies the different policy with which it learnt. For example - It learns by epsilon greedy policy but interacts with environment by greedy policy. 
It's like a person who learns from expert but applies his skills while actually riding the bike. 

\textbf(Q-Learning)
Q(S_t, A_t) = Q(S_t, A_t) + \alpha ((R_t + argmax_a Q(S_{t+1}, A_{t+1})) - Q(S_t, A_t))


Google Colab link for Cliff Walking Problem using SARSA: [link](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1psaWVCd2VKTG9YMzVNQnVXTVpWU1hDb01OQXxBQ3Jtc0tuUG81Mm8xR0RHdlRRU0tIOTJjaVRyOTNZQ1JRUGlKemtVY2RBSHd6NUpBbWQ3eFktcWhDcDFkV3hBTUxLQlhmdGdhTDg5Y2Y0RHM3eUgzU18wdzFQRGFmbUVEbDBLQ1MwVjRTdHhwUFd3MFNLWlU2UQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1XbxqH9eZ6r-TFO6wgQUahICDSHnHbXwN%3Fusp%3Dsharing&v=PfCME1G7hKI)

Google Colab link for Cliff Walking Problem using Q-Learning: [link](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXdTbjE4RFBJaG5GeDBxV25rbFU2U0s4Wm1NZ3xBQ3Jtc0tsUnY3d3pEbkJXSHpFajJyMGl0M09YLVRvQ28wT1VGUVJXMXpPTkNGbGpqdE1nbmFsZGhXMXFvcUd1NGtSamhRM1N3Q3VhVkFaeFVjMm1KTElCemgyeENIcFBMYUJvS0NnVDFJaTA5cXdfSkRSMFBtdw&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1NaBEZW-pSg8Uezumz1TA2peH1pVjq5qu%3Fusp%3Dsharing&v=PfCME1G7hKI)