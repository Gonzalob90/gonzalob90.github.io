---
title: "Dissertation: Non-Synergistic VAE"
date: 2019-01-01
tags: [generative modeling, variational inference]
excerpt: "Generative Modeling, Variational Inference"
mathjax: "true"
---

# H1 heading

## H2 heading

### H3 heading

Basic text

What about a [link](https://google.com)

<img src="{{ site.url }}{{ site.baseurl }}/images/population.png" alt="population coding">

$$\require{physics}$$

$$ S_{max}(\{X_{1},X_{2},...,X_{n}\};Y) = I(\bm{X}; Y) - \sum_{y \in Y} p(Y=y) \max_{i} KL \big[\ P(A_{i} | y) \Vert P(A_{i}) \big]\ $$

$$ I(\mathbb{A}_{i};Y=y) = \sum_{a_{i} \in \mathbb{A}_{i}} P(a_{i} | y) \log  \frac{P(a_{i},y)}{P(a_{i})P(y)} 
                  = KL \big[\ P(\mathbb{A}_{i} | y) \Vert P(\mathbb{A}_{i}) \big]\ $$
                  
```python
def i_max(indices, mu, log_var):

    mu_syn = mu[:, indices]
    log_var_syn = log_var[:, indices]
    i_max = kl_div(mu_syn, log_var_syn)

    return i_max
```

$$\mathcal{L}_{elbo}(\theta,\phi,x) =  E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}(x | z) \big]\ - KL \big[\ q_{\phi}(z | x) \Vert p(z) \big]\$$


$$\mathcal{L}_{new}( \theta,\phi,x ) =  \underbrace{E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}( x | z ) \big]\ - KL \big[\ q_{\phi}( z | x) \Vert p (z)\big]\ }_{\mathcal{L}_{elbo}}- \underbrace{\alpha KL \big[\ q_{\phi}(\mathbb{A}_{w} | x) \Vert p(\mathbb{A}_{w})\big]\ }_{\alpha*\text{Imax}}$$

<img src="{{ site.url }}{{ site.baseurl }}/images/nips_latents.png" alt="results">