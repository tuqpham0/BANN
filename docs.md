
## Currently Supported Divergences
For $\mathbb{R}^{\text{dim}}$, the supported divergences are
   - Generalized Kullback--Leibler (KL) divergence $D_{KL}(q_m\| \bullet)$
      - Domain: $D, Q \subset \mathbb{R}^{\text{dim}}_+$
      - If the domain is restricted to $\triangle^{d-1} = \{x\in \mathbb{R}^{\text{dim}}\,:\,\sum_{i=1}^{\text{dim}}x_i, x_i > 0\}$, then this reduces to the Kullback--Leibler divergence.
   - Reversed Kullback--Leibler divergence $D_{KL}(q_m \| \bullet)$
      - Domain: $D, Q \subset \mathbb{R}^{\text{dim}}_+$.
   - Itakura--Saito (IS) divergence $D_{IS}(q_m\|\bullet)$
      - Domain: $D, Q \subset \mathbb{R}^{\text{dim}}_+$.
   - Reversed Itakura--Saito divergence $D_{IS}(\bullet\|q_m)$
      - Domain: $D, Q \subset \mathbb{R}^{\text{dim}}_+$
   - Squared Euclidean (SE) distance $D_{SE}(q\|\bullet)$
      - Domain: $D, Q \subset \mathbb{R}^{\text{dim}}$

# (Approximate) Nearest Neighbour Search
#### Example usage
```
D = np.random.rand(20, 10)
Q = np.random.rand(15, 10)
nn_idx = bann.k_search(data = D, query = Q, k = 3, eps = 0, div = 'kl')
print(nn_idx)
# [[18 19  2]
# [17 14  2]
# [ 2 14 16]
# [12  2  3]
# [17 14 19]
# [14 13 17]
# [ 2 16 17]
# [19  5  2]
# [ 0  3 19]
# [ 0 14  2]
# [ 5  1 16]
# [14  6 17]
# [ 6  2 12]
# [18 19  2]
# [ 2  9  4]]
```
#### Overview
Given a set $D = \{d_{n}\}_{n=1}^{N}$ and a set $Q = \{q_{m}\}_{m=1}^{M}$, For each $q_m$, we compute the $k$-nearest neighbours in $D$ measured by either $D_{F}(q_m\|\bullet)$ or $D_F(\bullet \|q_m)$.

For the Squared Euclidean distance, it is known that the search is performed in $O(N\log N)$ for each query. The SE distance is symmetric, so we do not have to consider the reversed direction of computation. For other divergences, the complexity is unknown.
#### Parameters
   - **Data**: *numpy.ndarray*
      - 2 dimensional np.ndarray of size $(|D|,$ dimension$)$.
   - **Query**: *numpy.ndarray*
      - 2 dimensional np.ndarray of size $(|Q|,$ dimension$)$.
   - **k**: *int*, optional
      - Number of nearest neighbours to be computed for each query. Must have $0< k \le |D|$. Default value is $k=1$.
   - **eps**: *float*, optional
      - Error bound for search. Returned nearest neighbours are at most $(1+\epsilon)$ times further than true nearest neighbours. Default value is eps$=0$, which corresponds with exact nearest neighbours.
   - **div**: *str*, optional
      - Choice of divergence. Default value is div = 'kl'. Currently accepted inputs are:
         - 'kl'   :: KL divergence
         - 'dkl'  :: Reverse KL divergence
         - 'is'   :: IS divergence
         - 'dis'  :: Reverse IS divergence
         - 'se'   :: SE distance
#### Return
   - **nn_indices**: *numpy.ndarray*
      - 2 dimensional array of size $(|Q|, k)$. The $(i,j)$ entry will be the index for the $j^{th}$ nearest neighbour for the $i^{th}$ query point.
# (Approximate) Bregman--Hausdorff divergence
#### Example Usage
```
P = np.random.rand(20, 10)
Q = np.random.rand(50, 10)
bhaus_div = bann.bhaus(setp = P, setq = Q, eps = 0, div = 'kl')
print(bhaus_div)
# 0.841948848849059
```
#### Overview
Given two sets of vectors $P,Q$, computes the Bregman--Hausdorff divergence from $P$ to $Q$; $H_{D_{F}}(P\|Q)$
#### Parameters
   - **P**: *numpy.ndarray*
      - 2 dimensional np.ndarrray of size $(|P|, \text{dimension})$
   - **Q**: *numpy.ndarray*
      - 2 dimensional np.ndarray of size $(|Q|, \text{dimension})$
   - **eps**: *float*, optional
      - Error bound for search. Returned value is at most $(1+\epsilon)H_{D_F}(P\|Q)$. Default value is *eps*$=0$, corresponding to computing the exact Bregman--Hausdorff divergence from $P$ to $Q$.
   - **div**: *str*, optional
      - Choice of divergence. Default value is div = 'kl'. Currently accepted inputs are:
         - 'kl'   :: KL divergence
         - 'dkl'  :: Reverse KL divergence
         - 'is'   :: IS divergence
         - 'dis'  :: Reverse IS divergence
         - 'se'   :: SE distance
#### Return
   - **bhaus**: *float*
      - The Bregman--Hausdorff divergence from $P$ to $Q$; $H_{D_{F}}(P\|Q)$

# Test functions
#### Overview
The following functions are here to test various aspects of the functions.
##### Timing functions
The following functions will return the wall times for 
   - reading data sets into ANN
   - computing kd-tree
   - performing kd-tree search for NN or BH
```
bann.__timed_k_search(data, query, k = 1, eps = 0, div = 'kl')
bann.__timed_bhaus(P, Q, k = 1, eps = 0, div = 'kl')
```