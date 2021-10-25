<!-- .slide: data-background-color="#0f132d" data-background="./img/blue_galactic_neural_nets.jpg" -->
<h2 class="title">Advanced PG. RL applications in ML</h2>



### Contents
</br>

1. Recap++
2. Advanced PG
3. RL applications in ML
4. Practical advice
5. Open problems



<!-- .slide: data-background-color="#0f132d" data-background="./img/alpha_star.jpg" -->
<h2 class="title">AlphaStar: Grandmaster level in StarCraft II</h2>

<ul class="has-dark-background">
    <li class="fragment">$10^26$ actions at each step!</li>
    <li class="fragment">Vast space of strategies...</li>
    <li class="fragment">... not discoverable with naive exploration</li>
    <li class="fragment">Imperfect information</li>
    <li class="fragment">Planning horizon over thousands of steps</li>
</ul>



### So how did they do it?
</br>

1. Learn a policy with <span class="alert">supervized learning</span> (84%)
2. <span class="alert">Self-play</span> in a League of main agents and exploiters...
3. ... conditioned on human strategies.

</br>
<div class="fragment">
    <p>
        The RL part uses an <span class="alert">Actor-Critic</span>:
        <ul>
            <li class="fragment">with a $\text{TD}(\lambda)$ critic</li>
            <li class="fragment"><span class="alert">importance sampling </span> for off-policy correction</li>
        </ul>
    </p>
</div>



<!-- .slide: data-background-color="#fff" data-background="./img/league_training.png"  data-background-size="90%"  -->



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">Recap</h2>



### Value-based methods
</br>

<div class="small">
<ul>
    <li>Episode: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, ...$</li>
    <li class="fragment">Return: 
    $$
    \begin{aligned}
        G_t & \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\cdots \\
            & = R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\gamma^{2} R_{t+4}+\cdots\right) \\
            & = R_{t+1}+\gamma G_{t+1} \end{aligned}
    $$
    </li>
    <li class="fragment"> And we care about:
    $$
    \begin{aligned}
        v_{\pi}(s) 
            & \doteq \mathbb{E}_{\pi}[G_t | S_t=s] \\
            & = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t=s] \\
            & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s]
    \end{aligned}
    $$
    </li>
    <li class="fragment">TD error: $\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{improved estimate}} - \underbrace{V(S_t)}_{\text{current estimate}}$</li>
</ul>
</div>



### Approximation 
</br>

- We can approximate the <span class="alert">(action-)value</span> function:
`$$
\begin{aligned}
    V_{\theta}(s) & \approx V^{\pi}(s) \\ Q_{\theta}(s, a) & \approx Q^{\pi}(s, a)
\end{aligned}
$$`



#### One-step Temporal Difference VFA

<img src="./img/one_step_td.png" alt="Monte Carlo" width="600px">

`$$
\mathbf{\theta} \leftarrow 
    \mathbf{\theta}+\alpha\left[R_t+\gamma V_{\theta}(S_{t+1})-V_{\theta}(S_t)\right] \nabla V_{\theta}(S_t)
$$`



#### Monte Carlo VFA

<img src="./img/monte_carlo.png" alt="Monte Carlo" width="600px">

`$$
    \mathbf{\theta} \leftarrow 
        \mathbf{\theta}+\alpha \left[ G_{t} - V_{\theta}(S_t) \right] \nabla V_{\theta}(S_t)
$$`



### Deep Value-based RL
</br>

- Neural Fitted Q-learning
- Deep Q-Networks
    - Overestimation
    - Disambiguation 
    - Prioritization
    - Distributional perspective, auxiliary cost functions

<p class="fragment">Watch out for the <span class="alert">deadly triad</span>.</p>



#### Dynamic Programming

<img src="./img/dynamic_programming.png" alt="Dynamic Programming" width="600px">

<p class="small">
$$
    V(S_t) \leftarrow
        \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma V(S_{t+1}) \right] = \sum_{a} \pi(a | S_t) \sum_{s', r} p(s', r | S_t, a)[r + \gamma V(s')]
$$
</p>



<!-- .slide: data-background-color="#fff" data-background="./img/unified_view.png" data-background-size="40%" -->



#### Just Parametrize the Policy!
</br>

<ul>
    <li>Discrete:
        $$
            \pi(a | s, \boldsymbol{\theta}) \doteq \frac{e^{h(s, a, \boldsymbol{\theta})}}{\sum_{b} e^{h(s, b, \boldsymbol{\theta})}}
        $$
    </li>
    <li class="fragment">Continuous:
        $$
            \pi(a | s, \boldsymbol{\theta}) \doteq \mathcal{N}(\mu_{\theta}, \sigma^2_{\theta})
        $$
    </li>
</ul>



#### Objective
</br>

For performance measure:

`$$
    J(\boldsymbol{\theta}) \doteq v_{\pi_{\boldsymbol{\theta}}}\left(s_{0}\right),
$$`

be able to compute:

`$$
    \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \nabla_{\theta} J(\boldsymbol{\theta}).
$$`

<p class="fragment">A policy can be optimized with <span class="alert">non-gradient</span> methods also!</p>
<p class="fragment">The score function does not need to be <span class="alert">continuous</span>!</p>



### Why Policy Gradients?

- Effective in <span class="alert">high-dimensional</span> and <span class="alert">continuous</span> actions spaces
- Easier to approximate?
- Can learn <span class="alert">stochastic</span> policies
- Better convergence properties

<span class="cite">(<span class="alert">David Silver's</span>, lecture)</span>



### General case
</br>

<span class="alert">Score function</span> gradient estimator:

<small>
$$
\begin{aligned}
    \nabla_{\theta} \mathbb E_{x \sim p(x \mid \theta)} [f(x)]
    &= \nabla_{\theta} \sum_x p(x \mid \theta) \; f(x) & \text{expected value} \\
    & = \sum_x \nabla_{\theta} p(x \mid \theta) \; f(x) & \\
    & = \sum_x p(x \mid \theta) \frac{\nabla_{\theta} p(x \mid \theta)}{p(x \mid \theta)} \; f(x) \\
    & = \sum_x p(x \mid \theta) \nabla_{\theta} \log p(x \mid \theta) \; f(x) & \text{because: } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
    & = \mathbb E_x \left[ \; f(x) \nabla_{\theta} \log p(x \mid \theta) \right] & \text{take expectation}
\end{aligned}
$$
</small>



### Intuition
</br>

<img src="./img/pg.png" alt="Policy Gradient intuition" width="700px">
<small class="cite"><span class="alert">Karpathy</span>, 2016</small>



#### Monte-Carlo PG. Aka REINFORCE
</br>

`$$
    \nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ G_t } \right]
$$`

</br>

<pre><code class="hljs" data-trim data-line-numbers="1-9">
pi(a | s, w)        # parametrized policy
alpha > 0           # step size
w = randn((D,))     # weights

# tau = trajectory s0, a0, r1, ..., s_T-1, a_T-1, r_T ~ pi
for each tau ~ pi(. | ., w):
    for each t in tau:
        G_t = sum(r_t':T)
        w += alpha * G_t * grad(log(pi(a_t | s_t, w)))
</code></pre>



#### Baselines
</br>

`$$
    \nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; (G_t \color{#BD1421}{ -b(s_t) }) \right]
$$`

As long as the baseline <span class="alert">does not depend </span> on $a_t$ it can be shown:

- that the estimator remains unbiased
- that it will reduce the variance of the estimator



#### Actor-Critic Methods
</br>

<p class="small">
Alternate forms:
$$
\begin{aligned} \nabla_{\theta} J(\theta)
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ G_t } \right] & \text { REINFORCE } \\
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ Q^{\phi}(s, a) } \right] &  \text { Q Actor-Critic } \\
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ A^{\phi}(s, a) } \right] & \text { Actor-Critic } \\
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ \delta } \right] &  \text { TD Actor-Critic } \\
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \; \color{#BD1421}{ \delta e } \right] &  \operatorname{TD}(\lambda) \text { Actor-Critic }
\end{aligned}
$$
</p>



#### Actor-Critic methods
</br>

Actor-Critic methods maintain <span class="alert">two sets</span> (or more) of parameters:

- <span class="alert">Critic</span> updates (action-)value function parameters $\phi$
- <span class="alert">Actor</span> updates policy parameters $\theta$ in the direction suggested by the critic.



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">Advanced Policy Gradients</h2>



### Problems with PG
</br>

- Hard to choose <span class="alert">stepsize</span>
    - exacerbated by the data being nonstationary.
    - small change in parameters leads to large change in policy.
- PG methods are <span class="alert">online</span>, leading to worse sample efficiency 



### TRPO. Optimization perspective

- In PG we optimize the loss:
`$$
L^{P G}(\theta)=\mathbb{E}_{t}\left[\log \pi_{\theta}\left(a_{t} | s_{t}\right) A_{t}\right]
$$`

- Equivalently:
`$$
L_{\theta_{\mathrm{old}}}^{I S}(\theta) = 
    \mathbb{E}_{t}\left[\frac{\pi_{\theta}\left(a_{t} | s_{t}\right)}
    {\pi_{{\theta}_{\mathrm{old}}} \left(a_{t} | s_{t}\right)} A_{t}\right]
$$`



<!-- .slide: data-background-color="#fff" -->
#### Importance sampling
</br>

<img class="clean" src="./img/importance_sampling.png" alt="Importance Sampling" width="700px">

Can we evaluate <span class="alert">target</span> policy $\pi$ on data collected by <span class="alert">behaviour</span> policy $\mu$?

<small class="cite"><span class="alert">Munos</span>, 2018</small>



<!-- .slide: data-background-color="#fff" -->
#### Importance sampling
</br>

<img class="clean" src="./img/importance_sampling.png" alt="Importance Sampling" width="700px">

`$$
J(\theta) = 
    \mathbb{E}_{s \sim \rho^{\mu_{\theta}}, a \sim \mu_{\theta}} \big[ \frac{\pi_\theta(a \vert s)}{\mu_{\theta}(a \vert s)} A_{\mu}(s, a) \big]
$$`



#### Trust Region Policy Optimization

- Trust region update:
`$$
\begin{aligned}
    \text{maximize}_{\theta} & \quad L_{\theta_{\mathrm{old}}}^{I S}(\theta)=\mathbb{E}_t \left[\frac{\pi_{\theta}\left( a_t | s_t \right)}{\pi_{\theta_{\mathrm{old}}}\left( a_t | s_t \right)} A_t \right] \\
    \text{subject to}        & \quad \mathbb{E}_t \left[\mathrm{KL}\left[\pi_{\theta_{\mathrm{old}}}\left(\cdot | s_{t}\right), \pi_{\theta}\left(\cdot | s_{t}\right)\right]\right] \leq \delta
\end{aligned}
$$`

- Efficient way to compute the gradients of these two terms which include <span class="alert">2nd order derivatives</span>.



#### TRPO. Taylor expansion
</br>

<div class="small">
Loss:
$$
L_{\theta_{\mathrm{old}}}(\theta) = L_{\theta_{\mathrm{old}}}(\theta) + \mathbf{g}^T(\theta - \theta_{\mathrm{old}})
$$
Constraint:
$$
\mathrm{KL}[\pi_{\mathrm{old}}, \pi] = \mathrm{KL}[\pi_{\mathrm{old}}, \pi_{\mathrm{old}}] + 
                                       \nabla \mathrm{KL}[\pi_{\mathrm{old}}, \pi](\theta - \theta_{\mathrm{old}})
                                       + \frac{1}{2}(\theta - \theta_{\mathrm{old}})^T \mathbf{H}(\theta - \theta_{\mathrm{old}})
$$
Thus:
$$
\begin{aligned}
\theta_{t+1}  = & \; \text{argmax}_{\theta} \; \mathbf{g}^T(\theta - \theta_{\mathrm{old}}) \\
                & \; \text{s.t.}  \frac{1}{2}(\theta - \theta_{\mathrm{old}})^T \mathbf{H}(\theta - \theta_{\mathrm{old}}) \le \delta
\end{aligned}
$$

</div>



#### TRPO. The search direction
</br>

<div class="small">
We have the optimization problem:
$$
\begin{aligned}
\theta_{t+1}  = & \; \text{argmax}_{\theta} \; \mathbf{g}^T(\theta - \theta_{\mathrm{old}}) \\
                & \; \text{s.t.}  \frac{1}{2}(\theta - \theta_{\mathrm{old}})^T \mathbf{H}(\theta - \theta_{\mathrm{old}}) \le \delta
\end{aligned}
$$

Lagrange multiplier:
$$
G = \mathbf{g}^T \mathbf{s} - \lambda \frac{1}{2} \mathbf{s}^T \mathbf{H} \mathbf{s}
$$

<p class="fragment">
Differentiate w.r.t. $\mathbf{s}$ and set to 0:
$$
\frac{\partial G}{\partial \mathbf{s}} = \mathbf{g} - \lambda \mathbf{H}\mathbf{s} = 0
$$
Direction is given by solving $\mathbf{H}\mathbf{s} = \mathbf{g}$.
</p>

</div>



#### TRPO. Sum-up:
</br>

<div class="small">
    <ul>
        <li>Purely optimization formulation</li>
        <li class="fragment">Solve a constraint optimization problem so that $\pi, \pi_{\text{old}}$ stay close during an update</li>
        <li class="fragment">Do a <span class="alert">linear</span> approximation to $L(\theta)$ and <span class="alert">quadratic</span> to the $\text{KL}$ constraint</li>
        <li class="fragment">Use <span class="alert">conjugate gradient</span> to get an optimization direction: $\mathbf{s} \sim \mathbf{H}^{-1} \mathbf{g}$</li>
        <li class="fragment">Compute the step $\beta = \sqrt{\frac{2\delta}{\mathbf{s}^T\mathbf{Hs}}}$</li>
    </ul>
</div>




#### Proximal Policy Optimization
</br>

- Force the <span class="alert">importance sampling ratio</span> to stay within $[1-\varepsilon, 1+\varepsilon]$
`$$
L^{CLIP}(\theta) = \mathbb{E}_{t}\left[ \min \left[ r_t(\theta)A_t, \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) A_t \right] \right]
$$`
- This avoids extreme policy updates.
- The value function is trained as usual, with TDE.



<!-- .slide: data-background-color="#fff" -->
#### IMPALA
</br>

<img class="clean" src="./img/impala.png" alt="IMPALA" width="700px">

Remember A3C is essentially online. IMPALA needs to deal with off-policy data.



<!-- .slide: data-background-color="#fff" -->
#### IMPALA V-Trace
</br>

<img class="clean" src="./img/importance_sampling.png" alt="Importance Sampling" width="700px">

<div class="small">
$$
v_s \doteq V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) 
    \underbrace{\rho_t \left( r_t + \gamma V(x_{t+1}) - V(x_t) \right)}_{\delta_tV}
$$
with $\rho_i = \text{min}\left( \overline{\rho}, \, \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)} \right)$ and 
$c_i = \text{min}\left( \overline{c}, \, \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)} \right)$.
</div>



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">RL applications in ML</h2>



#### Recurrent models of Attention
</br>

<img class="clean" src="./img/ram.png" alt="Show and Tell" width="500px">



<!-- .slide: data-background-color="#fff" data-background="./img/ram_example.png" data-background-size="60%" -->



#### Optimizing a non-differentiable metrics
</br>

<img class="clean" src="./img/show_and_tell.png" alt="Show and Tell" width="700px">

<span class="alert">Problem: </span> the differentiable losses are unnatural. The newer (better) such as SPICE are not.



<!-- .slide: data-background-color="#fff" data-background="./img/spice_algo.png" data-background-size="50%" -->



#### Subpixel Neural Tracking
</br>

<img class="clean" src="./img/tracking_axons.png" alt="Show and Tell" width="700px">

Policy + Baseline. Clever $R(s)$ function.



<!-- .slide: data-background-iframe="https://distill.pub/2016/augmented-rnns/" data-background-interactive -->



#### Dialog Systems
</br>

<img class="clean" src="./img/dialogue_systems.png" alt="Show and Tell" width="700px">

<span class="cite">Steve Young, 2017</span>



#### Practical advice
</br>

<div class="small">
    <ul>
        <li>Log everything.</li>
        <li class="fragment">No, <span class="alert">really</span>, log everything:
        <ul>
            <li>Maximum Q-value</li>
            <li>TD-error</li>
            <li>Gradient magnitude</li>
            <li>Entropy, auxiliary losses</li>
            <li>Episodic return, reward / step</li>
            <li>Mean steps / episode</li>
            <li>FPS rate</li>
            <li>Pay attention to performance</li>
        </ul>
        </li>
        <li class="fragment">Always have a <span class="alert">distinct evaluation routine</span> that runs a couple hundred eval episodes.</li>
        <li class="fragment">A good idea is to also keep a <span class="alert">cache of evaluation episodes</span>.</li>
        <li class="fragment">Start small, find the simplest env that illustrates the problem.</li>
        <li class="fragment">At least three seeds or go home.</li>
        <li class="fragment">Log everything, question everything.</li>
    </ul>
</div>



<!-- .slide: data-background-color="#fff" data-background="./img/reproducibility.jpeg" data-background-size="60%" -->



#### Open Problems
</br>

<ul>
    <li class="fragment">"Exploration, exploration, exploration!" - Precup, 2019
    <ul>
        <li>pseudo-counts</li>
        <li>intrinsic motivation</li>
        <li>maximum entropy policies</li>
    </ul>
    </li>
    <li class="fragment">Hierarchical learning</li>
    <li class="fragment">Generalization</li>
    <li class="fragment">Continual learning</li>
    <li class="fragment">Model-based?</li>
</ul>




# Questions?
