# PROTAX-py
Python implementation of [PROTAX](https://pubmed.ncbi.nlm.nih.gov/27296980/) 

# Functionality
Estimates the probability of each outcome in a taxonomic tree given a query sequence compared to reference sequences. The probability vector corresponding to all branches under a node, $z$, is:

$$P(z) = \frac{\mathbf{\mathbf{w} \odot \exp(X\beta)}}{\mathbf{\exp(X\beta)\mathbf{w}}}$$

Probability of an assigned outcome is:

$$P(outcome | \beta,q) = q*prior+(1-q)\prod_{z_i \in path} P(z_i)_i$$

### Todo
- fix notation
# Requirements
- Python >=3.9
- NumPy 1.23.1