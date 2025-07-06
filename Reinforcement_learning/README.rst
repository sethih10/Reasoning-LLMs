===============================
Reinforcement Learning Basics
===============================

Policy
------

* Defines the behaviour of the agent and its interaction with the environment.

Reward Signal
-------------

* Maximises short‑term success.

Value Function
--------------

* Maximises long‑term success.

Model
-----

* If the environment is complex we sometimes build mathematical models to
  simulate the agent–environment system.


Multi‑arm Bandit Problem
========================

The multi‑arm bandit is a toy example of complex reinforcement‑learning
problems.  It is modelled as a slot‑machine game with several levers: pulling
one particular lever has the best probability of winning, but we must discover
which one without any initial knowledge of the system’s state.

Value‑action method
-------------------

In this method the agent interacts with the environment and receives a reward
for each action.  We observe the long‑run average of the value–action pair for
each lever to estimate the probability of choosing the best lever.

:math:`q_i(a)=1` if you win  
:math:`q_i(a)=0` if you lose

Over many interactions (law of large numbers) we estimate the most probable
winner.  A purely greedy search can get stuck in a local optimum, so we balance
**exploration** and **exploitation**:

:math:`q_i(a)=\frac{1}{t}\sum_{k=1}^t q_{i,k}(a)`


Markov Decision Process
=======================

Agent–Environment Interface
---------------------------

* **State** – describes the system (temperature, energy level, …).
* **Action** – immediate operation chosen by the agent.
* **Policy** – mapping from state to action, :math:`\pi(a\mid s)`.

Rewards and Expected Returns
----------------------------

* **Reward** :math:`r_t` – immediate feedback at time‑step *t*.
* **Return** :math:`G_t = r_t + r_{t+1} + \dots + r_T`.

Task types
~~~~~~~~~~

* **Episodic** – interaction stops at a known horizon (e.g. chess).
* **Continuous** – horizon unknown; discount future reward  
  :math:`G_t=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k}`, where
  :math:`0<\gamma<1`.

Markov Property
---------------

:math:`P(S_{t+1},R_t\mid S_t,A_t)
      = P(S_{t+1},R_t\mid S_0,A_0,\dots,S_t,A_t)`

Only the current state matters.

Markov Decision Processes
-------------------------

Any RL task with the Markov property is a **Markov Decision Process (MDP)**.


Value Functions
===============

State‑value function
--------------------

:math:`v_{\pi}(s)=\mathbb{E}_{\pi}\!\left[G_t\mid S_t=s\right]
  =\mathbb{E}_{\pi}\!\left[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k}\mid S_t=s\right]`

Action‑value function
---------------------

:math:`q_{\pi}(s,a)=\mathbb{E}_{\pi}\!\left[G_t\mid S_t=s,A_t=a\right]`

Bellman equation
----------------

:math:`v_{\pi}(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}
  P(s',r\mid s,a)\,[\,r+\gamma\,v_{\pi}(s')\,]`

Optimal value and policy
------------------------

:math:`v_{*}(s)=\max_{a}\sum_{s',r}P(s',r\mid s,a)
  \,[\,r+\gamma\,v_{*}(s')\,]`


Bellman Derivation
==================

.. image:: Bellman_derivation.png
   :alt: Derivation


Dynamic Programming
===================

Prediction Problem
------------------

*Policy evaluation*:

:math:`v_{\pi}(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
  \,(r+\gamma\,v_{\pi}(s'))`

Iterative update:

:math:`v_{k+1}(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
  \,(r+\gamma\,v_{k}(s'))`

Control Problem
---------------

*Policy optimisation* (value iteration):

:math:`v_{k+1}(s)=\max_{a}\sum_{s',r}p(s',r\mid s,a)
  \,(r+\gamma\,v_{k}(s'))`

Dynamic programming requires known transition probabilities.


Monte‑Carlo Methods
===================

When transition probabilities are unknown we estimate them via interaction.

On‑policy vs off‑policy
-----------------------

* **On‑policy** – behaviour policy = target policy.  
* **Off‑policy** – behaviour policy ≠ target policy (importance sampling).

Interactive examples:  
`On‑policy MC control <https://claude.ai/public/artifacts/1e531122-1b45-4337-86a2-df8bc2ac531b>`_  
`Importance‑sampling ratio <https://claude.ai/public/artifacts/8a773548-d360-418b-a20b-d427dc7a1952>`_


Temporal‑Difference Learning
============================

Combines DP bootstrapping with MC sampling → no need for full episode.

Prediction Problem
------------------

MC update:

:math:`V_{k+1}(s)=V_k(s)+\alpha\,[\,G_k-V_k(s)\,]`

TD(0) update:

:math:`V(s_t)\leftarrow V(s_t)+\alpha\,[\,r_{t+1}
        +\gamma\,V(s_{t+1})-V(s_t)\,]`

Control Problem
---------------

**SARSA** (on‑policy):

:math:`Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\,[\,r_{t+1}
        +\gamma\,Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\,]`

**Q‑learning** (off‑policy):

:math:`Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\,[\,r_{t+1}
        +\gamma\max_{a}Q(s_{t+1},a)-Q(s_t,a_t)\,]`

Cliff‑walking notebooks:  
`SARSA <https://colab.research.google.com/drive/1XbxqH9eZ6r-TFO6wgQUahICDSHnHbXwN>`_ •
`Q‑learning <https://colab.research.google.com/drive/1NaBEZW-pSg8Uezumz1TA2peH1pVjq5qu>`_


Function Approximation Methods
==============================

In large state‑spaces we approximate :math:`v_{\pi}` with parameters
:math:`w`:

:math:`w_{t+1}=w_t-\alpha\,[\,v_{\pi}(s_t)-\hat{v}(s_t,w_t)\,]
   \nabla\hat{v}(s_t,w_t)`

Semi‑gradient TD:

:math:`w_{t+1}=w_t-\alpha\,[\,r_{t+1}
   +\gamma\,\hat{v}(s_{t+1},w_t)-\hat{v}(s_t,w_t)\,]
   \nabla\hat{v}(s_t,w_t)`

Prediction and Control
----------------------

* Linear methods  
* Non‑linear methods – neural networks (Deep RL)

Control uses the same TD target with :math:`q(s,a,w)`.


Policy‑Gradient Methods
=======================

We parameterise the policy: :math:`\pi(a\mid s,\theta)` and maximise
performance :math:`J(\theta)` with gradient ascent:

:math:`\theta_{t+1}=\theta_t+\alpha\,\nabla J(\theta)`

Policy‑Gradient Theorem
-----------------------

:math:`\nabla J(\theta)\propto
  \sum_s \mu(s)\sum_a q_{\pi}(s,a)\,\nabla\pi(a\mid s,\theta)`

REINFORCE
---------

:math:`\theta_{t+1}=\theta_t+\alpha\,
  G_t\,\frac{\nabla\pi(a_t\mid s_t,\theta)}{\pi(a_t\mid s_t,\theta)}`

REINFORCE with baseline:

:math:`\theta_{t+1}=\theta_t+\alpha\,
  (G_t-\hat{v}(s_t,w))\,\frac{\nabla\pi(a_t\mid s_t,\theta)}
                                {\pi(a_t\mid s_t,\theta)}`

Actor–Critic:

:math:`\theta_{t+1}=\theta_t+\alpha\,
  (r_{t+1}+\gamma\,\hat{v}(s_{t+1},w)-\hat{v}(s_t,w))\,
  \frac{\nabla\pi(a_t\mid s_t,\theta)}{\pi(a_t\mid s_t,\theta)}`


Generalised Advantage Estimation
================================

Advantage: :math:`A(s_t,a_t)=q_{\pi}(s_t,a_t)-v_{\pi}(s_t)`

n‑step, :math:`\lambda`‑weighted estimate leads to GAE:

:math:`\hat{A}(s_t,a_t)=
  (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} \bigl(
       \underbrace{r_t+\gamma r_{t+1}+\dots+\gamma^{n-1}r_{t+n-1}}_{n\text{-step}}
       +\gamma^{n}v_{\pi}(s_{t+n})-v_{\pi}(s_t)\bigr)`
