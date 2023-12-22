# Amortized bootstrap
**Implicit Bootstrap Model**. We define the amortized bootstrap by parametrizing the bootstrap distribution $F(\theta)$ with a differentiable function $f_{\phi}$ with parameters $\phi$. Samples are drawn from the bootstrap distribution via $\hat{\theta} = f_{\phi}(\xi)$ where $\xi \sim p_0$ with $p_0$ being some fixed distribution. In this paper, we take $f_{\phi}$ to be a neural network (NN). Thus, we define the bootstrap distribution $F(\theta)$ implicitly through the functional sampler $f_{\phi}$—a so-called implicit model (Mohamed & Lakshminarayanan, 2016). In other words, we cannot evaluate $F$ but we can sample from it easily, similarly to the generative component of a Generative Adversarial Network (GAN) (Goodfellow et al., 2014) or the posterior defined by variational programs (Ranganath et al., 2016).

**Algorithm 1: Amortized Bootstrap Learning**

**Input:** Max iterations $T$, learning rate $\eta$, number of replications $K$, dataset $X_0$, data model $p(x|\theta)$, sampler $f_{\phi}$, fixed distribution $p_0(\xi)$.

```
for t in range(1, T+1):
    X_1, ..., X_K ~ G(x)  # sample every few epochs
    ξ_1, ..., ξ_K ~ p_0(ξ)  # sample for each minibatch
    θ_{\phi;k} = f_{\phi}(\xi_k) for k in range(1, K+1)
    φ_t = φ_{t-1} + η * \frac{\partial J(X_0, \phi_{t-1})}{\partial \phi_{t-1}}
Return amortized bootstrap parameters $\phi_T$
```
Optimization Objective. The parameters $\phi$ can be estimated by optimizing the likelihood function under samples from the implicit model:

$$ J(X_0, \phi) = \mathbb{E}{F(\theta)}\mathbb{E}{G(x)} [\log p(X|\theta)] \approx \frac{1}{K} \sum_{k=1}^{K} \log p(X_k|\theta_{\phi;k}) \quad (3) $$

where $X_k ~ G(x)$ and $\theta_{\phi;k} = f_{\phi}(\xi^k)$. Notice that this objective is the same as that of the traditional bootstrap: if the $K$ models are independent, maximizing the sum is equivalent to maximizing each term individually.

Gradient-Based Optimization. Learning the amortized bootstrap distribution amounts to maximizing Equation 3 with respect to $\phi$. We assume the model parameters $\theta$ are continuous and thus can take gradients directly through the parameter samples and then into $f_{\phi}$ as follows:

$$ \frac{\partial J(X_0, \phi)}{\partial \phi} = \frac{1}{K} \sum_{k=1}^{K} \frac{\partial \log p(X_k|\theta_{\phi;k})}{\partial \theta_{\phi;k}} \frac{\partial \theta_{\phi;k}}{\partial \phi} \quad (4) $$

The optimization procedure is summarized in Algorithm 1. The user must specify how often to sample the data and parameters from $G(x)$ and $F(\theta)$ respectively. We found that sampling new parameters for every minibatch and sampling new datasets after every few epochs (3 to 5) works well.

Amortized Bagging. Using the amortized bootstrap for amortized bagging is straightforward. At test time, we draw some number of samples $M$ from $f_{\phi}$—$f_{\theta^1}, \ldots, f_{\theta^M} = f_{\phi}(\xi^m)$—and then we ensemble the $M$ models $p(X|\theta^m)$ via the usual mechanisms (voting, averaging, etc.). Here we explicitly see the benefits of using the amortized bootstrap: the size of the ensemble $M$ can be larger than the number of replications used during training. This is not possible with traditional bagging.
