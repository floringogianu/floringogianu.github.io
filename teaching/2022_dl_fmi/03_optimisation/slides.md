<!-- .slide: data-background-color="#F0F0F0" data-background="./img/hero.png" data-background-size="60%" data-background-position="bottom right" -->
<h2 class="title">Optimization for DNNs</h2>



<!-- .slide: data-background-color="#F0F0F0" data-background="./img/course_map.png" -->



### Contents
</br>

1. Recap++
2. Gradient descent
3. Problems with (S)GD
4. Advanced optimization methods



### Recap. Linear models
</br>

<img class="clean" src="./img/data_objective_space.png" alt="Linear Models. Objective Space." width="90%">

Find the weights either analytically or iteratively.



<!-- .slide: data-background-iframe="http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/" data-background-interactive -->



### Recap. Transformations
</br>

<ul>
    <li><code>$\bm{y} = \bm{W}\bm{x}$</code> computes 
        <span class="alert">rotation</span>, 
        <span class="alert">scale</span> and 
        <span class="alert">reflection</span> transforms,
    </li>
    <li><code>$\sigma(\bm{W}\bm{x})$</code> adds <span class="alert">non-linear</span> transforms</li>
</ul>

<div class="grid2x fragment">
    <img class="clean" src="./img/transform_input.png" alt="Gaussian Input.">
    <img class="clean" src="./img/transform_random_nn2.png" alt="Random NN transform.">
</div>



### Recap. Fully connected NN

<img src="./img/nn_arch.png" alt="MLP." width="80%">

- Linear case: <code>$f(\bm{x}) = \bm{W}\bm{x}$</code>
- Neural net: <code>$f(\bm{x}) = \bm{W}_3\sigma(\bm{W}_2 \sigma(\bm{W}_1\bm{x}))$</code>



### Recap. Fully connected NN

<img src="./img/multi_graph.png" alt="MLP explicit." width="70%">

- Neural net:    <code>$f(\bm{x}) = \bm{y} = \bm{W}_2 \sigma(\bm{W}_1\bm{x} + \bm{b}_1) + \bm{b}_2$</code>
<!-- - Loss function: <code>$\mathcal{L}(\bm{W}) = \sum_{i=0}^N (y_i - t_i)^2$</code> -->
- Loss function: <code>$\mathcal{L} = \frac{\bm{y} \cdot \bm{t}}{\lVert \bm{y} \rVert \lVert \bm{t} \rVert}$</code>



### Recap. Computation graphs & AD

<img src="./img/computation_graph_svm.png" alt="Computation Graph." width="80%">

<span class="alert">Automatic Differentiation:</span> takes a program which
computes a value, and automatically constructs a procedure for computing
derivatives of that value with respect to some inputs.

<span class="cite alert">Grosse, Ba, CS421</span>



### Recap. Backpropagation

<img src="./img/zoom4.png" alt="Computation Graph." width="80%">

<span class="cite alert">Fei-Fei et. al, CS231n</span>



### Recap. Backprop patterns

<div class="grid2x2">
    <div class="grid-item"><img class="clean" src="./img/pattern_add.png" alt="Addition gate."></div>
    <div class="grid-item"><img class="clean fragment" src="./img/pattern_mul.png" alt="Multiplication gate."></div>
    <div class="grid-item"><img class="clean fragment" src="./img/pattern_copy.png" alt="Copy gate."></div>
    <div class="grid-item"><img class="clean fragment" src="./img/pattern_max.png" alt="Max gate."></div>
</div>



### Recap. Backprop beyond scalars
</br>

<ul>
    <li>scalar to scalar: $x \in \mathbb{R}, y \in \mathbb{R}$. Derivative: $\frac{\partial y}{\partial x} \in \mathbb{R}$</li>
    <li class="fragment">vector to scalar: $\bm{x} \in \mathbb{R}^N, y \in \mathbb{R}$. 
        <strong>Gradient</strong>: $\frac{\partial y}{\partial \bm{x}} \in \mathbb{R}^N$
    </li>
    <li class="fragment">vector to vector: $\bm{x} \in \mathbb{R}^N, \bm{y} \in \mathbb{R}^M$.
        <strong>Jacobian</strong>: $\frac{\partial \bm{y}}{\partial \bm{x}} \in \mathbb{R}^{N \times M}$
    </li>
</ul>



### Recap. Some other points
</br>

<ul>
    <li>Can we compute $\frac{\partial y}{\partial x} \in \mathbb{R}$ with $y \sim \mathcal{N}(0, x)$?</li>
    <p class="fragment">(yes, sort of, check <a href="https://arxiv.org/abs/1906.10652">Mohamed, 2019</a>)</p>
    <li class="fragment">While <span class="alert">depth</span> is good, it can be problematic</li>
    <li class="fragment">Initialization matters!</li>
    <li class="fragment">Also, <span class="alert">Universal Approximation Theorem</span> does not imply every function is also <span class="alert">learnable</span>!</li>
</ul>



<!-- .slide: .centered data-background-color="#0f132d" -->
<h2 class="title">Optimization</h2>



### Gradient Descent
</br>

<div class="grid2x">
    <div style="max-width: 60%">
        <ul>
            <li>Measure the error of our model using $\mathcal{L}(\mathcal{D}, \bm{\theta})$,</li>
            <li>Compute the gradients w.r.t. $\theta$,</li>
            <li>Find incremental solutions that better explain the data using:</li>
        </ul>
        <div class="centered">
        $$\bm{\theta}_{j+1} \leftarrow \bm{\theta}_j - \eta \nabla_{\bm{\theta}_j}\mathcal{L}$$
        </div>
    </div>
    <img src="./img/gd.png" alt="Gradient Descent." style="max-width:40%;">
</div>



### Stochastic Gradient Descent
</br>

Consider instead objective functions that are the sum of the losses -- $\mathcal{L}(\bm{\theta}) = \sum_{n=1}^{N}\mathcal{L}_n(\bm{\theta})$, resulting in the update:

<div class="centered">
$$\bm{\theta}_{j+1} \leftarrow \bm{\theta}_j - \eta \sum_{n=1}^{N} \nabla_{\bm{\theta}_j}\mathcal{L}_n(\bm{\theta})$$
</div>

</br>
<ul>
    <li class="fragment">evaluating the sum of gradients can be expensive</li>
    <li class="fragment">instead we can randomly choose a subset of $\mathcal{L}_n$</li>
    <li class="fragment">for gradient descent to converge we only require the samples to be unbiased</li>
</ul>



<!-- .slide: data-background-color="#F0F0F0" data-background="./img/landscapes.png" data-background-size="80%" -->



### Taylor Approximation
</br>

- Arbitrary non-linear functions are hard to globally analyze.
- Ideally, we would like to work with simple functions: <span class="alert">polynomials</span>.
- <span class="aler">Solution</span>: get a polynomial that approximates the
function in a <span class="alert">neighbourhood</span> well enough

</br>
</br>
<div class="centered">$$
\mathcal{L}(\bm{\theta}) \approx
    \mathcal{L}\left(\bm{\theta}_{0}\right)+
    \nabla \mathcal{L}\left(\bm{\theta}_{0}\right)^{\top}\left(\bm{\theta}-\bm{\theta}_{0}\right)+
    \frac{1}{2}\left(\bm{\theta}-\bm{\theta}_{0}\right)^{\top} \mathbf{H}\left(\bm{\theta}_{0}\right)\left(\bm{\theta}-\bm{\theta}_{0}\right)
$$</div>



### Taylor Approximation
</br>

<img src="./img/taylor_approximation.png" style="max-width: 60%;" alt="Taylor series">

$\text{sin}(x)$ and its Taylor approximations, polynomials of degree
<span style="color:red;">1</span>,
<span style="color:orange;">3</span>,
<span style="color:yellow;">5</span>,
<span style="color:lime;">7</span>,
<span style="color:blue;">9</span>,
<span style="color:indigo;">11</span>, and
<span style="color:violet;">13</span>.
By <a href="https://commons.wikimedia.org/w/index.php?curid=27865201">IkamusumeFan, CC BY-SA 3.0</a>.



### First-Order

<div class="grid2x" style="max-width: 80%">
    <img src="./img/tangent_line.png" alt="Gaussian Input.">
    <img src="./img/tangent_space.png" alt="Random NN transform.">
</div>

Growth of a function when a) changing $\theta$ and b) moving in the direction of $\theta - \theta_0$.

<div class="centered">$$
\mathcal{L}(\bm{\theta}) \approx
    \mathcal{L}\left(\bm{\theta}_{0}\right)+
    \nabla \mathcal{L}\left(\bm{\theta}_{0}\right)^{\top}\left(\bm{\theta}-\bm{\theta}_{0}\right)
$$</div>



### Critical points
</br>

<img src="./img/critical_points.png" style="max-width:90%;" alt="Critical Points">

<div class="centered">$$
\mathcal{L}(\bm{\theta}) \approx
    \mathcal{L}\left(\bm{\theta}_{0}\right)+
    \frac{1}{2}\left(\bm{\theta}-\bm{\theta}_{0}\right)^{\top} \mathbf{H}\left(\bm{\theta}_{0}\right)\left(\bm{\theta}-\bm{\theta}_{0}\right)
$$</div>

How does moving in a direction affect the rate of change.



### Patologies. Plateaux

<div class="grid2x" style="max-width: 80%">
    <img src="./img/plateau.png" style="max-width:50%;" alt="Critical Points">
    <img src="./img/tanh.png" style="max-width:50%;" alt="Tanh Activation">
</div>

<ul>
    <li>In cases where a neuron is <span class="alert">saturated</span>:
        </br>$\delta_{\bm{\theta}} = \delta_{\bm{z}} \bm{x}$,
        </br>$\delta_{\bm{z}} =\delta_{\bm{h}} \frac{\partial\bm{h}}{\partial \bm{z}} = \delta_{\bm{h}} \phi^{\prime}(\bm{z})$.</li>
    <li class="fragment"> Solutions: use ReLU activations and good initialization schemes.</li>
</ul>



### Patologies. Bad curvature

<div class="grid2x" style="max-width: 80%">
    <img src="./img/good_conditioning.png" style="max-width:39%;" alt="Good">
    <img src="./img/bad_conditioning.png" style="max-width:61%;" alt="Bad">
</div>

<p>
<span class="aler">Remeber</span>: $\bm{\theta}_{j+1} \leftarrow \bm{\theta}_j - \alpha \nabla_{\bm{\theta}_j}\mathcal{L}$
</p>

Badly conditioned curvatures arise even in simple cases, when the inputs have
slightly different scales or are off-centered. 

<span class="alert">Solutions:</span> i) standardize your input data, ii)
normalize every pre-activation.



<!-- .slide: data-background-iframe="https://distill.pub/2017/momentum/" data-background-interactive -->



### Closed-form GD update
</br>

Consider the objective: 

<div class="centered">
$\mathcal{L}(\bm{\theta}) = \frac{1}{2} \bm{\theta}^{\intercal}\underbrace{(\bm{X}^{\intercal}\bm{X})}_{\bm{A}}\bm{\theta}$,
</div>

Write the gradient descent update as:

<div class="centered">
$$\begin{aligned}
\bm{\theta}_{t+1} & \leftarrow \bm{\theta}_{t}-\alpha \nabla \mathcal{L}\left(\bm{\theta}_{t}\right) \\
&=\bm{\theta}_{t}-\alpha \mathbf{A} \bm{\theta}_{t} \\
&=(\mathbf{I}-\alpha \mathbf{A}) \bm{\theta}_{t}
\end{aligned}$$
</div>

Breaking the recursion:

<div class="centered">
$$
    \bm{\theta}_t = (\mathbf{I} - \alpha \mathbf{A})^{t} \bm{\theta}_0.
$$
</div>



### Component-wise

Spectral decomposition: $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\intercal}$ so that we can do a change of basis:

<div class="centered">
$$
\begin{aligned}
(\mathbf{I}-\alpha \mathbf{A})^{t} \bm{\theta}_{0} &=\left(\mathbf{I}-\alpha \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{\top}\right)^{t} \bm{\theta}_{0} \\
&=\left[\mathbf{Q}(\mathbf{I}-\alpha \bm{\Lambda}) \mathbf{Q}^{\top}\right]^{t} \bm{\theta}_{0} \\
&=\mathbf{Q}(\mathbf{I}-\alpha \bm{\Lambda})^{t} \mathbf{Q}^{\top} \bm{\theta}_{0}
\end{aligned}
$$
</div>

In the eigenspace of $\mathbf{Q}$, each coordinate is multiplied by $(1 - \alpha \lambda_i)^{t}$

<span class="alert">Behaviours</span>:

- $0 \lt \alpha\lambda_i \leq 1$: decays to 0 at a rate depending on $\lambda_i$
- $1 \lt \alpha\lambda_i \leq 2$: oscillations
- $\alpha \lambda_i \gt 2$: diverges



### Learning dynamics

<img src="./img/typical_curve.png" style="max-width:50%;" alt="Good">

The optimizer is quickly descending along the direction of the largest eigenvalues.



### Fix #1. Momentum
</br>

Plain gradient descent:
<div class="centered">
$$\bm{\theta}_{t+1} \leftarrow \bm{\theta}_t - \alpha \nabla_{\bm{\theta}_t}\mathcal{L}$$
</div>

With <span class="alert">momentum</span>:
<div class="centered">
$$\begin{aligned}
\bm{v}_{t+1} &\leftarrow \mu \bm{v}_t + \nabla_{\bm{\theta}_t}\mathcal{L}\\
\bm{\theta}_{t+1} &\leftarrow \bm{\theta}_t - \alpha \bm{v}_{t+1}.
\end{aligned}$$
</div>

- $\mu$: <span class="alert">damping factor</span> and controls _how much momentum_ we use.
- Rolling ball on a slope analogy.
- Keeps the _historic_ direction in low curvature regions and _dampens_ oscillations in high curvature directions.



<!-- .slide: data-background-iframe="https://distill.pub/2017/momentum/" data-background-interactive -->



### Fix #1. Momentum
</br>

- no reason not to use it, it almost always helps...
- ...except when modelling distributions and small changes in $\theta$ means a large change in the distribution,
- low memory complexity,
- also check Nesterov momentum.



### Fix #2. RMSprop
</br>

Plain gradient descent:
<div class="centered">
$$\bm{\theta}_{t+1} \leftarrow \bm{\theta}_t - \alpha \nabla_{\bm{\theta}_t}\mathcal{L}$$
</div>

With <span class="alert">RMSprop</span>:
<div class="centered">
$$\begin{aligned}
    \bm{s}_t &\leftarrow (1 - \gamma) \bm{s}_t + \gamma [\nabla_{\bm{\theta}_t}\mathcal{L}]^{2}, \\
    \bm{\theta}_{t+1} &\leftarrow \bm{\theta}_t - \frac{\alpha}{\sqrt{\bm{s}_t + \epsilon}} \nabla_{\bm{\theta}_t}\mathcal{L}.
\end{aligned}$$
</div>

- adaptive learning rate!
- <span class="alert">small components</span> will use a higher step size
- alternate view: the gradient is rescaled to have norm 1



### Fix #3. Adam

Bring back <span class="alert">momentum</span>!

<div class="centered">
$$\begin{aligned}
\bm{v}_{t} &\leftarrow \mu \bm{v}_{t-1} + (1-\alpha) \nabla_{\bm{\theta}_t}\mathcal{L}\\
\bm{s}_t   &\leftarrow \gamma \bm{s}_{t-1} + (1 - \gamma) [\nabla_{\bm{\theta}_t}\mathcal{L}]^{2}, \\
\end{aligned}$$
</div>

The parameter update is then:

<div class="centered">
$$
\bm{\theta}_{t+1} \leftarrow \bm{\theta}_t - \frac{\alpha}{\sqrt{\bm{s}_t + \epsilon}} \bm{v}_t.
$$
</div>



### Comparison

<div class="grid2x" style="max-width: 100%">
    <img src="./img/contours_evaluation_optimizers.gif" style="max-width:50%;" alt="Good">
    <img src="./img/saddle_point_evaluation_optimizers.gif" style="max-width:50%;" alt="Bad">
</div>



### Regularization

<img class="clean" src="./img/optim_reg.png" style="max-width:80%;" alt="Regularization">



<!-- .slide: .centered data-background-color="#0f132d" -->
# Questions?