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

Python code block:
```python
    import numpy as np
    def test_function(x,y)
        z = np.sum(x,y)

```

Here's some math:
$$z= x+y$$
$$\newcommand{\infdiv}{D_{KL}\infdivx}$$
$$\require{physics}$$
$$\mathcal{L}_{new}( \theta,\phi,x ) =  \underbrace{E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}( x | z ) \big]\ - KL \big[ q_{\phi}( z | x) \Vert p (z)\big]\}_{\mathcal{L}_{elbo}}- \underbrace{\alpha KL \big[\{q_{\phi}(\mathbb{A}_{w} | x) \Vert p(\mathbb{A}_{w})\big]\ }_{\alpha*\text{Imax}}$$

