# Bregman Approximate Nearest Neighbours for Python
This is a python wrapper for the (Decomposable) Bregman Approximate Nearest Neighbour(Bregman ANN, or BANN) package adapted by Hubert Wagner and Tuyen Pham. The original ANN package was written by David Mount and Sunil Arya. The original package uses Kd-trees to search for nearest neighbours in Euclidean space equipped with an $L^{p}$-norm. [See David Mount's page for documention for the original ANN package.](https://www.cs.umd.edu/~mount/ANN/)

For a Python wrapped version of the original ANN, see:
   - [pyANN by Dmitriy Morozov](https://mrzv.org/software/pyANN/)
   - [PyANN by Anna Nevison](https://github.com/annacnev/pyann)

#### Bregman Divergences
Bregman divergences are measurements of generalized distances in a space. Unlike metrics, they are often assymmetric and do not globally satisfy the triangle inequality. Recently, these divergences have been useful in machine learning, with the most prominent example being the Kullback--Leibler divergence.

## About
The BANN package currently uses Kd-trees for two primary computations:
   - (approximate) $k$-nearest neighbour searches with decomposable Bregman divergences
   - Bregman--Hausdorff divergence for decomposable Bregman divergences

Currently, this package supports the following divergences:
   - Kullback--Leibler divergence (primal and dual)
   - Itakura--Saito divergence (primal and dual)
   - Squared Euclidean divergence

Additional decomposable divergences can be simply added to the source code, and passing a divergence from Python is a planned implementation.

### Details
Let $D = \{d_n\}_{n=1}^{N}$, $Q = \{q_m\}_{m=1}^{M}$, and $D_F$ be a decomposable Bregman divergence.
#### Bregman $k$-nn search
For $q\in Q$, the Bregman $k$-nearest neighbour search returns the ordered list of indices $({x_{1},x_{2},\dots,x_{k}})$, such that \[D_F(q\| d_{x_{1}}) \leq D_F(q\| d_{x_{n}}) \leq \cdots D_F(q\| d_{x_{k}}),\] and $D_F(q\|d_{x_k}) \leq D_F(q\| x_\ell)$ for all $\ell\notin\{x_{1},x_{2},\dots,x_{k}\}$. As Bregman divergences are rarely symmetric, we can reverse the arguments as necessary.

This package also supports $\epsilon$-approximate nearest neighbour searches, where the divergence to the reported nearest neighbour is at most $(1+\epsilon)$ times the divergence to the true nearest neighbour.

[Further details of using Kd-trees with Bregman--Divergences are discussed here.](https://arxiv.org/abs/2502.13425)
#### Bregman$-$Hausdorff divergence
The Bregman--Hausdorff divergence generalizes the Bregman divergence between two vectors to the Bregman divergence between to *sets* of vectors. The Bregman--Hausdorff divergence was introduced by Pham, Dal Poz Kouřimská, and Wagner, where they also provide algorithms for its computation. Specifically, we compute \[H_{D_{F}}(P\|Q) = \sup_{d\in D}\inf_{q\in Q} D_F(d\|q)\] and \[H'_{D_F}(P\|Q)=\sup_{d\in D}\inf_{q\in Q}D_F(q\|d)\] via the shell algorithm. Note that the directions of computations for the Bregman--Hausdorff divergences are reversed compared to the directions for the nearest neighbour searches.

[The Bregman--Hausdorff divergence and shell algorithm for computation are introduced here.](https://www.mdpi.com/2504-4990/7/2/48)
## Requirements
### Python Version
BANN requires Python >=3.6, and Python 3.11 is recommended.
### Dependencies
   - [Numpy](https://numpy.org/)
## Installation
## Feedback
Bug reports, pull requests after forking, and other questions may be sent to the maintainer: tuyen.pham@ufl.edu
## Copyright and License
