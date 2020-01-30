---
title: "Dissertation: Non-Synergistic VAE"
date: 2019-01-01
tags: [generative modeling, variational inference]
excerpt: "Generative Modeling, Variational Inference"
mathjax: "true"
---

# H1 heading

### H3 heading

Basic text

What about a [link](https://google.com)

<img src="{{ site.url }}{{ site.baseurl }}/images/population.png" alt="population coding">

The intuition behind this metric is that synergy should be defined as the "whole beyond the 
maximum of its parts". The whole is described as the mutual information between the joint $\textbf{X}$ 
and the outcome Y; whereas the maximum of all the possible subsets is interpreted as the maximum information 
that any of the sources $\sA_{i}$ provided about each outcome. Formally, this is stated as:

$$\require{physics}$$

$$ S_{max}(\{X_{1},X_{2},...,X_{n}\};Y) = I(X; Y) - \sum_{y \in Y} p(Y=y) \max_{i} KL \big[\ P(A_{i} | y) \Vert P(A_{i}) \big]\ $$

$$\begin{align}
I(\mathbb{A}_{i};Y=y) &= \sum_{a_{i} \in \mathbb{A}_{i}} P(a_{i} | y) \log  \frac{P(a_{i},y)}{P(a_{i})P(y)} \\
                  &= KL \big[\ P(\mathbb{A}_{i} | y) \Vert P(\mathbb{A}_{i}) \big]\                   
\end{align}$$
                  
```python
def i_max(indices, mu, log_var):

    mu_syn = mu[:, indices]
    log_var_syn = log_var[:, indices]
    i_max = kl_div(mu_syn, log_var_syn)

    return i_max
```

$$\mathcal{L}_{elbo}(\theta,\phi,x) =  E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}(x | z) \big]\ - KL \big[\ q_{\phi}(z | x) \Vert p(z) \big]\$$

$$\begin{align}
\mathcal{L}_{new}(\theta,\phi,x) &= \frac{1}{N}\sum^{N}_{i=1} \bigg[\ E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}(x^{(i)} | z) \big]\ \bigg]\ - KL \big[\ q_{\phi}(z_{n}) \Vert p(z_{n}) \big]\ - I(x_{n};z) \nonumber \\
& \underbrace{- \alpha I(x_{n};z)}_\text{Penalise} + \alpha \sum_{x \in X} p(X=x) \max_{i} KL \big[\ q_{\phi}(\mathbb{A}_{i} | x){p(\mathbb{A}_{i})}) 
\end{align}$$

$$\mathcal{L}_{new}( \theta,\phi,x ) =  \underbrace{E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}( x | z ) \big]\ - KL \big[\ q_{\phi}( z | x) \Vert p (z)\big]\ }_{\mathcal{L}_{elbo}}- \underbrace{\alpha KL \big[\ q_{\phi}(\mathbb{A}_{worst} | x) \Vert p(\mathbb{A}_{worst})\big]\ }_{\alpha*\text{Imax}}$$

<img src="{{ site.url }}{{ site.baseurl }}/images/nips_latents.png" alt="results">