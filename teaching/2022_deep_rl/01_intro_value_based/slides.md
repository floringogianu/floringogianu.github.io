<!-- .slide: data-background="./img/blue_galactic_neural_nets.jpg" -->
<h2 class="title has-dark-background">Introduction to Reinforcement Learning</h2>



### Resources
</br>

Textbooks:

- Sutton, _Reinforcement Learning: An Introduction_, 2nd edition
- Pineau, _An introduction to Deep RL_
- Szepesvari, _Algorithms for Reinforcement Learning_



### Resources
</br>

Online lectures:

- [Udacity, Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600)
- [DeepMind, Advanced DL and RL](https://github.com/enggen/DeepMind-Advanced-Deep-Learning-and-Reinforcement-Learning)
- [Silver, Introduction to RL](https://youtu.be/2pWv7GOvuf0)



<!-- .slide: data-background-iframe="https://spinningup.openai.com/en/latest/spinningup/keypapers.html" data-background-interactive -->



### Evaluation

<!-- <div class="grid2x2 half">
    <img src="./img/minihack_gameplay.gif" alt="" class="grid_item" width="207px">
</div> -->
<img src="./img/crafter_gameplay.gif" alt="">

<div class="small">
<a href="https://github.com/danijar/crafter">Crafter</a>
<!-- <a href="https://github.com/facebookresearch/minihack/blob/main/docs/envs/tasks.md">MiniHack</a> -->

You will need to provide a report containing:
- proper evaluation curves of the trained agents
- description of the algorithms
- interesting behaviours you observed
</div>



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">What is RL?</h2>



### Why Reinforcement Learning?
</br>

> <span class="alert">Intelligence</span> measures an agent's ability to achieve goals in a wide range of environments.

`$$
\xi(\pi)=\sum_{\mu \in \mathcal{M}} 2^{-K(\mu)} V_{\mu}^{\pi}
$$`

<small class="cite">(<span class="alert">Legg & Hutter</span>, 2007)</small>



<!-- .slide: data-background-color="#fff" data-background="./img/action_perception_loop.png" data-background-size="70%" -->



### Why Reinforcement Learning?
</br>

- **Supervised learning:** mapping from data to values or classes, requires
    <span class="alert">labeled</span> data.
- **Unsupervised learning:** mapping from data to 
    <span class="alert">interesting</span> patterns.



### Why Reinforcement Learning?
</br>

By contrast, RL differes in that:

- there is no labeled data, only a <span class="alert">reward</span> signal
- reward may be <span class="alert">delayed</span>
- it is essentially an <span class="alert">online</span> setting
- data is highly <span class="alert">correlated</span> in time
- data distribution <span class="alert">changes</span> depending on the actions



<!-- .slide: data-background="./img/covers.png" data-background-size="50%" -->


<!-- .slide: data-background-color="#fff" data-background="./img/alpha_go.png" data-background-size="30%" -->



<!-- .slide: data-background-color="#fff" data-background="./img/td_gammon.png" data-background-size="50%" -->



<!-- .slide: data-autoplay data-background-iframe="https://www.youtube.com/embed/kopoLzvh5jY" -->



<!-- .slide: data-background-color="#fff" data-background="./img/open_ended_world.png" data-background-size="70%" -->



<h3>How about RL in the <span class="alert">Real Life</span></h3>
</br>

- [Online advertising](https://vowpalwabbit.org/)
- Dialogue systems
- [Project Loon](https://blog.x.company/drifting-efficiently-through-the-stratosphere-using-deep-reinforcement-learning-c38723ee2e90)
- [Target COVID testing of travellers](https://www.nature.com/articles/d41586-021-02556-w)
- Data center energy optimization
- [RL for Real Life ICML workshop](https://sites.google.com/view/RL4RealLife#h.p_E8GavvJ-X7nT)
- Robotics, a good place to start is [Pieter Abbeel's page](http://people.eecs.berkeley.edu/~pabbeel/)



<!-- .slide: data-background-color="#fff" data-background="./img/alpha_tensor.svg" data-background-size="70%" -->



### What is RL?
</br>

<q>Reinforcement learning is learning what to do - how to map situations to
actions - so as to maximize a numerical reward signal.</q>
<small class="cite">(<span class="alert">Sutton</span>, 2019)</small>

<q>Science of learning to make decisions from interaction</q>
<small class="cite">(<span class="alert">van Hasselt</span>, 2019)</small>



<!-- .slide: data-background-color="#fff" data-background="./img/treasure_hunt.png" data-background-size="40%" -->



### Optimal policies
</br>

<div class="grid">
    <img src="./img/op_r0.png" alt="" class="grid_item">
    <img src="./img/op_r1.png" alt="" class="grid_item fragment">
    <img src="./img/op_r2.png" alt="" class="grid_item fragment">
    <img src="./img/op_r3.png" alt="" class="grid_item fragment">
</div>

<small class="cite fragment">(<span class="alert">Dragan</span>, 2019)</small>



### Concepts
</br>

- <span class="alert">Reward</span> $R_t$. Scalar signal, performance at step $t$.
- <span class="alert">Action</span> $A_t$. Action taken by the agent according to:
    - <span class="alert">Deterministic</span> policy $a_t = \pi(s_t)$.
    - <span class="alert">Stochastic</span> policy $a_t \sim \pi(a_t | s_t)$.
- <span class="alert">Observation</span> $O_t.$ High-dimensional vector, different from the environment state.
- <span class="alert">State</span> $S_t$. Internal state of the environment.



<!-- .slide: data-background-color="#fff" data-background="./img/action_perception_loop.png" data-background-size="70%" -->



### Return
</br>

Return: $G_t = R_{t+1} + R_{t+2} + R_{t+3} ... $

</br>
<span class="alert">Discounted</span> return:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

</br>
<p class="fragment">Recursive: $G_t = R_{t+1} + \gamma G_{t+1}$</p>



<h3><span class="alert">Reward hypothesis:</span></h3>

<q>Any goal can be formalized as the outcome of maximizing a cumulative reward.</q>

</br>
<p class="fragment">See also the General Value Function framework.</p>



### Value Function
</br>

`$$
    v^{\pi}(s) = \mathbb E_{\pi} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-1} R_{T} | S_t = s, \pi]
$$`

- The value depends on the policy $\pi$
- Value functions are used to evaluate the utility of an action 
- Or to select between actions.



<h3><span class="alert">Bellman</span> equation</h3>
</br>

Since the return $G_t$ has a recursive form,

`$$
\begin{aligned}
    v^{\pi}(s)
    & = \mathbb E_{\pi} [R_{t+1} + \gamma G_{t+1} | S_t = s, \pi] \\
    & = \mathbb E_{\pi} [R_{t+1} + \gamma v^{\pi}(s_{t + 1}) | S_t = s, \pi] \\
\end{aligned}
$$`

And this holds for the <span class="alert">optimal</span> value also:

`$$
    v^*(s) = \text{max}_a \mathbb{E} [R_{t+1} + \gamma v^*(s_{t + 1}) | S_t = s, a=a]
$$`



### Action-Value Function
</br>

`$$
    q^{\pi}(s,a) = \mathbb E_{\pi} [R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a, \pi]
$$`

It's related to the value function in that:
`$$
    v^{\pi}(s) = \mathbb{E}_{a \sim \pi} [q^{\pi}(s,a)]
$$`



$v^{\pi}(s), q^{\pi}(s,a), v^{\ast}(s), q^{\ast}(s,a)$ are <span
class="alert">theoretical</span> objects

While $V_{t}(s), Q_t(s,a)$ are their <span class="alert">estimates</span>.



### Policy Evaluation
</br>

`$$
\begin{aligned} v_{\pi}(s) & \doteq \mathbb{E}_{\pi}\left[G_{t} | S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) | S_{t}=s\right] \\ &=\sum_{a} \pi(a | s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r | s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right] \end{aligned}
$$`



### Iterative Policy Evaluation
</br>

Imagine a succession of approximate value functions $v_0, v_1, v_2$:

`$$
\begin{aligned} 
    v_{k+1}(s) 
        & \doteq \mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{k}\left(S_{t+1}\right) | S_{t}=s\right] \\
        &=\sum_{a} \pi(a | s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r | s, a\right)\left[r+\gamma v_{k}\left(s^{\prime}\right)\right]
\end{aligned}
$$`

$v_k$ <span class="alert">can be shown to converge as</span> $k \to \infty$



### Policy Improvement
</br>

Say we determined $v_{\pi}$ for some arbitrary policy. Should we change the policy?

Would we get a better $v_{\pi}$?

`$$
    \pi^{n e w}(. | s)=\arg \max _{a} Q^{\pi}(s, a)
$$`

It turns out this is <span class="alert">guaranteed</span> to be an improvement.



<!-- .slide: data-background-iframe="https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html" data-background-interactive -->



### Generalized Policy Iteration
</br>

<img src="./img/gpi.png" alt="Generalized Policy Iteration" width="700px">



### Dynamic Programming

<img src="./img/dp.png" alt="Dynamic Programming" width="600px">

`$$
V(s_t) \leftarrow \sum_{a} \pi(a | s_t) \sum_{s^{\prime}, r} p\left(s^{\prime}, r | s_t, a\right)\left[r+\gamma V\left(s^{\prime}\right)\right]
$$`



### Monte Carlo

<img src="./img/monte_carlo.png" alt="Monte Carlo" width="600px">

`$$
V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]
$$`



### One-step Temporal Difference or TD(0)

<img src="./img/one_step_td.png" alt="Monte Carlo" width="600px">

`$$
V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]
$$`



### Unified View
</br>

<img src="./img/unified_view.png" alt="Unified View" width="500px">



### MC vs TD

- Monte Carlo is gradient-based, converges to a local optimum of the weighted MSE
- Temporal-Difference Learning (TD) fits (conceptually) a Markovian model
- Approximates the Value Function that would be computed from the model
- TD introduces further bias in order to reduce variance
- Bias comes from Markov assumption, use of current (incorrect) value estimates

<span class="cite">(<span class="alert">Precup</span>, 2019)</span>



<!-- .slide: data-background-iframe="https://distill.pub/2019/paths-perspective-on-value-learning/" data-background-interactive -->



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">TD(0) methods for control</h2>



### On-Policy Control
</br>

<small><span class="alert">SARSA</span> update:</small>

<small>
$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right)\right]
$$
</small>



### Off-Policy Control

<small><span class="alert">Q-learning</span> update: </small>
<small>
$$
Q(s_t, a_t) \leftarrow
    Q(s_t, a_t) + \alpha[R_{t+1} + \gamma \mathop{\arg\max}\limits_{a_{t+1}} Q(s_{t+1},a_{t+1}) - Q(s_t, a_t)]
$$
</small>

<small><span class="alert">Expected SARSA</span> update: </small>

<small>
$$
\begin{aligned} 
    Q\left(S_{t}, A_{t}\right) 
        & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \mathbb{E}_{\pi}\left[Q\left(S_{t+1}, A_{t+1}\right) | S_{t+1}\right]-Q\left(S_{t}, A_{t}\right)\right] \\
        & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \sum_{a} \pi\left(a | S_{t+1}\right) Q\left(S_{t+1}, a\right)-Q\left(S_{t}, A_{t}\right)\right]
\end{aligned}
$$
</small>



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">So, is RL done?</h2>



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">Value Function Approximation</h2>

<img class="clean" src="./img/function_approx.png" alt="Value Function Approximation" width="700px">




### VFA Objective
</br>

`$$
\overline{\mathrm{VE}}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s)\left[v_{\pi}(s)-\hat{v}(s, \mathbf{w})\right]^{2}
$$`

- minimize the good old <span class="alert">Mean Squared Error</span>,
- weighted by the state distribution $\mu(s)$.
- SGD will converge to a <span class="alert">local</span> minimum,
- Linear VFA has only one local minimum.



### SGD
</br>

`$$
\begin{aligned}
    \mathbf{w}_{t+1}
        & \doteq \mathbf{w}_{t}-\frac{1}{2} \alpha \nabla\left[v_{\pi}\left(S_{t}\right)-\hat{v}\left(S_{t}, \mathbf{w}_{t}\right)\right]^{2} \\
        &=\mathbf{w}_{t}+\alpha\left[v_{\pi}\left(S_{t}\right)-\hat{v}\left(S_{t}, \mathbf{w}_{t}\right)\right] \nabla \hat{v}\left(S_{t}, \mathbf{w}_{t}\right)
    \end{aligned}
$$`

This is <span class="alert">guaranteed</span> to converge as $\alpha \rightarrow 0$ under some conditions. 

<p class="fragment">But we generaly <span class="alert">don't know</span> $v_{\pi}(S_{t})$!</p>



### Monte Carlo VFA

<img src="./img/monte_carlo.png" alt="Monte Carlo" width="600px">

`$$
    \mathbf{w} \leftarrow \mathbf{w}+\alpha\left[G_{t}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right] \nabla \hat{v}\left(S_{t}, \mathbf{w}\right)
$$`



### One-step Temporal Difference VFA

<img src="./img/one_step_td.png" alt="Monte Carlo" width="600px">

`$$
\mathbf{w} \leftarrow \mathbf{w}+\alpha\left[R+\gamma \hat{v}\left(S^{\prime}, \mathbf{w}\right)-\hat{v}(S, \mathbf{w})\right] \nabla \hat{v}(S, \mathbf{w})
$$`

<p class="fragment">This is not a <span class="alert">full (true) gradient</span> anymore!</p>



### Semi-Gradient Methods
</br>

<div class="small">
$$
\mathbf{w} \leftarrow \mathbf{w}+\alpha\left[U_t-\hat{v}(S, \mathbf{w})\right] \nabla \hat{v}(S, \mathbf{w})
$$

- If $U_t$ is an <span class="alert">unbiased</span> estimate of $v^{\pi}(S_t)$, such as $G_t$, all good!

- When bootstrapping however: $U_t = R+\gamma \hat{v}\left(S^{\prime}, \mathbf{w}\right)$
    - the <span class="alert">true gradient</span> update is: $\mathbf{w} \leftarrow \mathbf{w}+\alpha\left[U_t-\hat{v}(S, \mathbf{w})\right] (\nabla \hat{v}(S, \mathbf{w}) - \gamma \nabla \hat{v}(S_{t+1}, \mathbf{w}))$.
    - the target depends on the current $\mathbf{w}$.

</div>

</br>
<span class="alert fragment">
Semi-gradient takes into account the effect of changing $\mathbf{w}$ on the estimate but ignore its effect on the target.
</span>



### Deadly triad (I)
</br>
</br>

- Function approximation
- Bootstrapping
- Off-policy training

<span class="cite">(<span class="alert">Sutton</span>, 2019 [11.3])</span>



### Before NN Approximators
</br>

Take a linear approximator:

`$$
\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^{\top} \Phi(s) \doteq \sum_{i=1}^{d} w_{i} \phi_{i}(s)
$$`

and compose it with some <span class="alert">non-linear</span> feature extractor $\Phi(s)$.

<p class="fragment">How about convergence? Gordon, 1999, shows Watkins & Dayan's results holds only for certain $\Phi(s)$.</p>



### Coarse coding example
</br>

<img src="./img/coarse_coding.png" alt="DQN" width="600px">



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">... Needs more layers</h2>



### Neural Fitted Q-Learning
</br>

- Dataset $\mathcal{D} = \{<s, a, r, s^{\ast}>\}$
- Build target: $Y_k^{Q} = r + \gamma \max_{a^{\ast}} Q(s^{\ast}, a^{\ast}; \theta_k)$

Minimize the loss:
`$$
\mathcal{L}_{\text{NFQ}} = \left[Q(s, a; \theta_k) - Y^{Q}_k \right]^2
$$`

<p class="fragment"><span class="alert">Problem: </span>Updating $\theta$ also updates the target.</p>



### Deep Q-networks (II)
</br>

<img src="./img/action_perception_loop_er.png" alt="DQN" width="600px">

</br>
<p class="small">
$$
\mathcal{L}_{\text{DQN}}(\boldsymbol{\theta})=\mathbb{E}_{\left(s, a, r, s^{\prime}\right) \sim U(\mathcal{D})}
    \big[\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \boldsymbol{\theta}^{-}\right)}_{\text {refined estimate }}-\underbrace{Q(s, a ; \boldsymbol{\theta})}_{\text {current estimate }})^{2}\big]
$$
</p>



### Deep Q-networks (II)

- introduces a separate <span class="alert">target network</span>
- adapts <span class="alert">Experience Replay </span> from (<span class="alert">Lin</span>, 1992)
- last four frames are grayscaled and stacked
- rewards are clipped to [-1, 1]
- loss of life = terminal state
- actions are repeated four times
- optimized with RMSprop

<p class="fragment">Huge leap over previous linear estimators with handcrafted feature extractors.</p>



<!-- .slide: data-background-color="#0f132d" data-background="./img/atari_domain.png" data-background-size="50%" -->



<!-- .slide: data-background-color="#fff" data-background="./img/rainbow.png" data-background-size="40%" -->



### Action Overestimation

- random estimation errors
- the approximator is not expressible
- non-stationarity of the data
- the $\texttt{max}$ operator is <span class="alert">positively biased</span>



### Maximization bias (I)
</br>

Suppose $Q(s, a; \theta) + \epsilon$ with $\mu=0$:

<img class="clean" src="./img/ddqn_noise.png" alt="ddqn_noise">

the target can be overestimated up to

`$$
    \gamma\epsilon\frac{m-1}{m+1}
$$`



### Maximization bias (II)
</br>

<img class="clean" src="./img/overestimation_tabular.png" alt="overestimation" width="600px">

</br>
Even in tabular settings!



### Double Q-learning

- Set two estimates, $Q_1(a), Q_2(a)$ and use them to:
    - determine the maximizing action $A^{\ast} = \text{argmax}_{a} Q_1(a)$,
    - estimate its value $Q_2(A^{\ast}) = Q_2(\text{argmax}_{a} Q_1(a))$,
    - then randomly switch roles.

- In Double _Deep_ Q-Networks however:

`$$
Y_{k}^{D D Q N}=r+\gamma Q\left(s^{\prime}, \underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s^{\prime}, a ; \theta_{k}\right) ; \theta_{k}^{-}\right)
$$`



<!-- .slide: data-background-color="#fff" data-background="./img/ddqn_results.png" data-background-size="60%" -->



### Dueling DQN (I)

- from the definition:

`$$
    V^{\pi}(s) = \mathbb E_{a \sim \pi(s)} \,[Q^{\pi}(s,a)]
$$`

- derive the <span class="alert">advantage</span> some action has with respect to $V(s)$

`$$
    A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
$$`

- that is, for optimal action $a^{\ast}$: $A^{\pi}(s, a^{\ast}) = 0$



### Dueling DQN (II)
</br>

<img class="clean" src="./img/dueling_dqn.png" alt="dueling_dqn" width="400px">

- Naively:
`$$
    Q(s,a; \theta, \alpha, \beta) = V(s;\theta, \beta) + A(s,a;\theta, \alpha)
$$`



### Dueling DQN (III)

- However we can't recover unique V and A so we force $A(s, a) = 0$ for the action taken:

`$$
\begin{aligned}
    Q(s,a; \theta, \alpha, \beta) = & V(s;\theta, \beta) \\
        & + (A(s,a;\theta, \alpha) - \mathop{\max}\limits_{a'\in |\mathcal A|} A(s,a';\theta, \alpha))
\end{aligned}
$$`

 - In practice:

`$$
\begin{aligned}
 Q(s,a; \theta, \alpha, \beta) = & V(s;\theta, \beta) \\
        & + (A(s,a;\theta, \alpha) - \frac{1}{|\mathcal A|}\Sigma_{a'}A(s,a';\theta, \alpha))
\end{aligned}
$$`



<!-- .slide: data-background-color="#fff" data-background="./img/dueling_saliency.png" data-background-size="40%" -->



### Prioritized Experience Replay

- Instead of sampling transitions <span class="alert">uniformly</span> from the ER...
- Sample those experiences from which the agent <span class="alert">would learn the most</span>.
- Specifically, sample $<s,a,r,s^{\ast}>$ tuples with a large absolute TD-error:

$$
\left|\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \boldsymbol{\theta}^{-}\right)-Q(s, a ; \boldsymbol{\theta})\right)^{2}\right|
$$

<p class="fragment">Can we do better than $|\text{TD-error}|$?</p>



### Multi-step learning

Recall the <span class="alert">n-step </span> methods discussed earlier. We can change the target so that we bootstrapp <span class="alert">later in the future</span>:

`$$
Y_{k}^{Q, n}=\sum_{t=0}^{n-1} \gamma^{t} r_{t+k+1}+\gamma^{n} \max _{a^{\prime} \in A} Q\left(s_{n}, a^{\prime} ; \theta_{k}\right)
$$`



### Distributional RL
</br>

<img class="clean" src="./img/dist_actions.png" alt="distributional_actions" width="600px">

</br>

- Why learn point estimates of the value-functions?
- Why not learn a <span class="alert">distribution</span> of returns?



### Distributional Bellman operator
</br>

<img class="clean" src="./img/dist_bellman.png" alt="distributional_bellman" width="600px">

</br>

<ul>
    <li class="fragment">define a <span class="alert">value distribution</span>: $Q^{\pi}(s, a)=\mathbb{E} Z^{\pi}(s, a)$</li>
    <li class="fragment">which is also recursive: $Z^{\pi}(s, a)=R\left(s, a, S^{\prime}\right)+\gamma Z^{\pi}\left(S^{\prime}, A^{\prime}\right)$</li>
    <li class="fragment">distributional loss: $D_{\mathrm{KL}}\left(\Phi {\mathcal{T}} Z_{\theta^-}(s, a) \| Z_{\theta}(s, a)\right)$</li>
</ul>



### Yeah, but why?
</br>

<img class="clean" src="./img/dist_why.png" alt="distributional_but_why" width="700px">



### Simpler is better?

<img class="clean" src="./img/rainbow_vs_spectral_norm.png" alt="rainbow_vs_spectral_norm" width="700px">

<div class="small">Optimisation rivals algorithmic improvements. Using spectral
normalization on a single layer of a C51 agent surpasses the performance of
Rainbow. Average over 54 Atari games (<a href="https://arxiv.org/pdf/2105.05246.pdf">paper</a>).
</div>



##### Auxiliary Tasks
<img class="clean" src="./img/unreal.png" alt="unreal" width="700px">



### Deadly triad (II)
</br>
</br>

Hypothesis:
- Deep divergence
- Target networks
- Overestimation
- Multi-step
- Capacity
- Prioritization

<span class="cite">(<span class="alert">van Hasselt</span>, 2018)</span>



#### Open questions

- RL is fairly unstable and difficult to optimise
- RL is sample inefficient
- How to include human priors about credit assignment?



# Questions?