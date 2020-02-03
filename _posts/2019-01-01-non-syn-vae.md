---
classes: wide
title: "Non-Synergistic Variational Auto-Encoders"
date: 2019-01-01
tags: [generative modeling, variational inference]
excerpt: "Generative Modeling, Variational Inference"
mathjax: "true"
---

### Work presented at NeurIPS 2018, Workshop LatinX in AI

Github (Reproduce results): [Link](https://github.com/Gonzalob90/non-synergistic-vae)

Our world is hierarchical and compositional, humans can generalise better since we use primitive
concepts that allow us to create complex representations (Higgins et al. (2016)).  
Towards the creation of truly intelligent systems, they should learn in a similar way resulting in an 
increase of their performance since they would capture the underlying factors of variation of the data 
( Bengio et al.(2013); Hassabis et al. (2017)).  

According to Lake et al. (2016), a compositional representation should create new elements from 
the combination of primitive concepts resulting in a infinite number of new representations.  For 
example if our model is trained with images of white wall and then is presented a boy with a white 
shirt, it should identify the color white as a primitive element. 

Furthermore, a disentangled representation has been interpreted in different ways, for instance Bengio et al. (2013) 
define it as one where single latent variables are sensitive to changes in generative factors, while being 
invariant to changes in other factors.  

The original Variational auto-encoder (VAE) framework (Kingma & Welling (2013); Rezende et al.(2014)) has been used 
extensively for the task of disentanglement by modifying the original ELBO formulation; for instance β-VAE, presented in Higgins et al. (2017a), 
increases the latent capacity by penalising the KL divergence term with a β hyperparameter.

$$\mathcal{L}_{elbo}(\theta,\phi,x) =  E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}(x | z) \big]\ - KL \big[\ q_{\phi}(z | x) \Vert p(z) \big]\$$

$$\mathcal{L}_{\beta-vae}(\theta,\phi,x) =  E_{q_{\phi}(z | x)} \big[\ \log p_{\theta}(x | z) \big]\ - \beta * KL \big[\ q_{\phi}(z | x) \Vert p(z) \big]\$$

We proposed a new approach for the task of disentanglement, where we achieve state-of-the-art results by penalizing the synergistic mutual 
information within the latents we encourage information independence and by doing that disentangle the latent factors. It is worth noting that our model draws inspiration
from population coding, a field from neuroscience, where the notion of synergy arises.

## Information Theory

First of all, we need a brief introduction to the field of information theory in order to understand the partial information decomposition, which it will lead us 
to the importance of Synergy in our framework. First, we define the Mutual Information:

$$\begin{equation}
I(X;Y) = E_{p(x,y)} \bigg[\ \log \frac{p(x,y)}{p(x)p(y)}  \bigg]\ = K \big[\ p(x,y) \Vert p(x)p(y) \big]\
\end{equation}$$

Now, we are going to introduce the Partial Information decomposition; following the notation from Williams & Beer (2010), let's consider the random variable S and a random vector $$R = \{R_{1}, R_{2}, .., R_{n}\}$$, being our goal to 
decompose the information that the variable R provides about S; the contribution of these partial information 
could come from one element (i.e. from $$R_{1}$$) or from subsets of $$R$$ (ie. $$R_{1}, R_{2}$$). Considering the case with 
two variables, $$\{R_{1}, R_{2}\}$$, we could separate the partial mutual information in unique ($$Unq(S; R_{1})$$ 
and $$Unq(S; R_{2})$$ ), the information that only $$R_{1}$$ or $$R_{2}$$ provides about S; redundant ($$Rdn(S; R_{1}, R_{2})$$), 
which it could be provided by $$R_{1}$$ or $$R_{2}$$; and synergistic ($$Syn(S; R_{1}, R_{2})$$), which is only 
provided by the combination of $$R_{1}$$ and $$R_{2}$$. This decomposition is displayed in the equation below:

$$\begin{equation}
I(S; R_{1},R_{2}) = \underbrace{Rdn(S;R_{1},R_{2})}_\text{Redundant} + \underbrace{Unq(S;R_{1})}_\text{Unique} + \underbrace{Unq(S;R_{2})}_\text{Unique} + \underbrace{Syn(S;R_{1},R_{2})}_\text{Synergistic}
\end{equation}$$

Below, we can see the representation of this decomposition for two variables; intuitively $$I(S; R_{1})$$ is the mutual information between the 
random vector S and the predictor $$R_{1}$$

<img src="{{ site.url }}{{ site.baseurl }}/images/2_variables.png" alt="results" class="align-center">

For three predictors we should point out that also is not only between individual predictions, but it can be also synergistic predictors, 
such as{1}{23}.  For the synergistic information, we should consider any information that is not carried by a single predictor, 
which can be redundant as well; for instance{12}{13} is synergistic.  As we can see below in the decomposition with 3 variables, as the number of predictors increases, 
the combinations increases dramatically. In this case, $$I(S; R_{1}, R_{2})$$ is the mutual information between the 
random vector S and the predictors $$R_{1}, R_{2}$$

<img src="{{ site.url }}{{ site.baseurl }}/images/synergy_3_latens.png" alt="results" class="align-center">

## Synergy

Synergy arises in different fields, being a popular notion of it as how much the whole is greater than the sum of its parts. The canonical example is the 
XOR gate (image below) where we need X1 and X2 to fully specify the value of Y. This mean that the mutual information provided jointly by X1 and 
X2 will provide1 bit of information, whereas the mutual information provided by any of the predictors will result in 0 bits.

<img src="{{ site.url }}{{ site.baseurl }}/images/xor.png" alt="results" class="align-center">

Below, we display the truth tables:

<img src="{{ site.url }}{{ site.baseurl }}/images/tables.png" alt="results" class="align-center">

### Population Coding

From the Neuroscience perspective, we know that single neurons make a small contribution to the 
animal behaviour.  Since most of our actions involve a large number of neurons, most of neurons 
in  the  same  region  will  have  a  similar  response,  which  is  why  the  ”population  coding”  field  is 
interested in the relationship between the stimulus and the responses of a group of neurons.  As 
this branch is related to the interactions of a group of neurons, it’s easy to find a close relation 
with Information Theory. Below we see a diagram that represent how a pair of neurons 1 and 2 encode information about 
a stimulus $$s_{t}$$ with the responses (spike trains) $$r_{1}(t)$$ and $$r_{2}(t)$$.  Broadly speaking, the spike trains 
are the representation of the stimulus experimented.  The process of encoding could be described 
as the conditional probability of the responses given the stimulus $$p(r_{1},r_{2}|s)$$.  On the other hand,the decoding 
process uses the neural spike trains to estimate the features or representation of the 
original stimulus.  Most  of  the  current  research  in  the  field  of  Information  Theory  was inspired  from  population coding.  In fact, it’s common to use metrics such as mutual information to measure the correlation 
between stimulus and response, $$I(S;R_{1},R_{2})$$.

<img src="{{ site.url }}{{ site.baseurl }}/images/population.png" alt="population coding">

For neural codes there are three types of independence when it comes to the relation between stimuli
and responses; which are the activity independence, the conditional independence and the information 
independence.  One of the first measures of synergy for sets of sources of information came from 
this notion of independence. In Williams & Beer (2010) it is stated that if the responses come 
from different features of the stimulus, the information encoded in those responses should be added 
to estimate the mutual information they provide about the stimulus. Formally:

$$I(S; R_{1}, R_{2}) = I(S;R_{1}) + I(S;R_{2})$$

However, we just saw in the previous sections that the $$I(S;R_{1})$$ and $$I(S;R_{2})$$could be decomposed 
in their unique and redundant and synergistic terms. Intuitively, this formulation only holds if there 
is no redundant or synergistic information present; which means in the context of population coding 
that the responses encoded different parts of the stimulus. If the responses $$R_{1}$$ and $$R_{2}$$ convey more 
information together than separate, we can say we have synergistic information; if the information 
is less, we have redundant information. That’s the reason why in Gat & Tishby (1998), the synergy 
is considered as measure of information independence

$$\begin{equation}
Syn(R_{1}, R_{2}) = I(S; R_{1}, R_{2}) - I(S;R_{1}) - I(S;R_{2})
\end{equation}$$

### Synergy Metric

* n: Number of individual predictors $$X_{i}$$
* $$\mathbb{A}_{i}$$ : subset of individual predictors (ie. $$A_{i} = \{X_{1},X_{3}\}$$)
* **X**: Joint random variable of all individual predictors $$X_{1}X_{2}..X_{n}$$
* $$\{X_{1},X_{2},...,X_{n}\}$$: Set of all the individual predictors
* Y: Random variable to be predicted
* y: A particular outcome of Y.

The intuition behind this metric is that synergy should be defined as the "whole beyond the 
maximum of its parts". The whole is described as the mutual information between the joint X 
and the outcome Y; whereas the maximum of all the possible subsets is interpreted as the maximum information 
that any of the sources $$\mathbb{A}_{i}$$ provided about each outcome. Formally, this is stated as:

$$\begin{align}
  S_{max}(\{X_{1},X_{2},...,X_{n}\};Y) &= I(X; Y) - I_{max}(\{\mathbb{A}_{1}, \mathbb{A}_{2} .. \mathbb{A}_{n}\};Y)\\
                  &= I(X; Y) - \sum_{y \in Y} p(Y=y) \max_{i}I(\mathbb{A}_{i};Y=y)
\end{align}$$

As we know, we can express the mutual information (I(X,Y)) as a KL divergence:

$$ I(\mathbb{A}_{i};Y=y) = \sum_{a_{i} \in \mathbb{A}_{i}} P(a_{i} | y) \log  \frac{P(a_{i},y)}{P(a_{i})P(y)} = KL \big[\ P(\mathbb{A}_{i} | y) \Vert P(\mathbb{A}_{i}) \big]\ $$

Combining the last two equations above, we define our Synergy metric as:

$$ S_{max}(\{X_{1},X_{2},...,X_{n}\};Y) = I(X; Y) - \sum_{y \in Y} p(Y=y) \max_{i} KL \big[\ P(\mathbb{A}_{i} | y) \Vert P(\mathbb{A}_{i}) \big]\ $$


## Model

Our hypothesis is that minimising synergistic information should encourage disentanglement of the factors of variation, we chose the 
overestimation of the synergy. First, we change the notation to match the VAE framework notation (Z is the joint of latents and X is the observations), 
the metric is the following:

$$S_{max}(\{Z_{1},Z_{2},...,Z_{d}\};X) = I(Z; X) - \sum_{x \in X} p(X=x) \max_{i} KL \big[\ P(\mathbb{A}_{i} | y) \Vert P(\mathbb{A}_{i}) \big]\ $$

Where $$A_{i}$$ is a subset of the latents, such that $$A_{i} \in \{Z_{1},Z_{2},...,Z_{n}\}$$ and Z is the joint of the latents. Formally: $$Z = \prod_{i}^{d} Z_{i}$$, where d is the number of latents.

Following the Variational Auto-encoder Framework, and assuming that the value of z is generated by a parametrised prior distribution $$p_{\theta}(z)$$ and x 
from the distribution $$p_{\theta}(x|z)$$, we recognise the intractability problem,  which is why we need to use an approximate distribution $$q_{\phi}(x | z)$$.
  
It is important to notice that this KL divergence could be computed in the same way as in the VAE framework; 
the only difference is the number of dimensions used for the random variable z.  In the  original  VAE  framework, 
we  compute  the  KL  divergence  considering  the  joint $$Z = \prod_{i}^{d} Z_{i}$$; whereas for the Synergy metric we don’t use the 
joint but a subset of the latents.  For instance, if $$A_{i}=Z_{2}Z_{5}Z_{8}$$, we have the following expression:

$$\begin{equation}
KL \big[\ q_{\phi}(z_{2}z_{5}z_{8} | x) \Vert p(z_{2}z_{5}z_{8}) \big]\
\end{equation}$$

We change the Synergy metric taking in account the intractability issue:

$$S_{max}(\{Z_{1},Z_{2},...,Z_{d}\};X) = I(Z; X) - \sum_{x \in X} p(X=x) \max_{i} KL \big[\ p(\mathbb{A}_{i} | y) \Vert q_{\phi}(\mathbb{A}_{i}) \big]\ $$


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

One of the ways to verify visually if the model disentangles the factors of variation is traversing 
the latent variables.  The process consists on traversing each dimension of the latent variable along 
five standard deviations around the unit Gaussian mean while keeping the rest of the dimensions 
constant.  This is consistent with the definition of disentanglement, which states that we should 
be able to change one factor of variation without affecting the other factors. On the left, we can see the traversal for this model after 1e6 steps.
Likewise,on the right we see the mean activation of each active latent averaged across shapes, rotations and scales.

<img src="{{ site.url }}{{ site.baseurl }}/images/traversal_mean_white.png" alt="results" class="align-center">

As we can see, our model is able to disentangle the latents (X axis, y axis, two rotation and scale); which matches with the results of state-of-the-art models such as 
B-VAE and Factor VAE.

## References:

* Irina Higgins, Loıc Matthey, Xavier Glorot, Arka Pal, Benigno Uria, Charles Blundell, Shakir Mohamed,  and Alexander Lerchner.  Early visual concept learning with unsupervised deep learning.CoRR, abs/1606.05579, 2016

* Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman.   Building machines that learn and think like people.CoRR, abs/1604.00289, 2016. URL http://arxiv.org/abs/1604.00289.

* Loic Matthey, Irina Higgins, Demis Hassabis, and Alexander Lerchner.  dsprites:  Disentanglement testing sprites dataset. https://github.com/deepmind/dsprites-dataset/, 2017.

* Diederik P. Kingma and Max Welling.   Auto-encoding variational bayes.CoRR, abs/1312.6114,2013. URL http://arxiv.org/abs/1312.6114.

* Yoshua Bengio, Aaron C. Courville, and Pascal Vincent.   Representation learning:  A review and new perspectives.IEEE Trans. Pattern Anal. Mach. Intell.,  35(8):1798–1828,  2013.

* Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra.   Stochastic backpropagation and approximate inference in deep generative models.  In Proceedings of the 31th International Conference on Machine Learning, ICML 2014. URL http://jmlr.org/proceedings/papers/v32/rezende14.html.

* Paul L. Williams and Randall D. Beer.   Nonnegative decomposition of multivariate information. CoRR, abs/1004.2515, 2010. URLhttp://arxiv.org/abs/1004.2515.9