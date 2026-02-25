import bann
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time

def compare_nns(inds1, inds2):
   x, y = len(inds1), len(inds2)
   z, w = len(inds1[0]), len(inds2[0])
   if x != y:
      print("Warning: different number of queries")
      return False
   if z != w:
      print("Warning: different number of neighbors between sets")
      return False
   else:
      for i in range(x):
         for j in range(z):
            if inds1[i][j] != inds2[i][j]:
               return False
      return True

np.random.seed(0)

P_size = 5_000
Q_size = 1_000

trials = 100
bann_t = np.zeros(trials)
sk_t = np.zeros(trials)

for dim in [1000]:
    for trial in range(trials):
        P = np.random.rand(P_size, dim)
        Q = np.random.rand(Q_size, dim)
#        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
#        print(f"Searching {Q.shape[0]} query points with |P| = {P.shape[0]} in dimension {dim}")
        start = time.time()
        bann_knn = bann.k_search(P, Q, k = 5, div = 'se')
        end = time.time()
        bann_t[trial] = end - start
#        print(f"Time taken for total BANN k-search: {end - start:.5f} seconds")

        start = time.time()
        full_kd = sk.neighbors.KDTree(P)
        fsk_inds = full_kd.query(Q, k = 5, return_distance = False)
        end = time.time()
#        print(f"Time taken for full build + k-search: {end - start:.5f} seconds")
        sk_t[trial] = end - start
        if not compare_nns(bann_knn, fsk_inds):
            print(f">>>>> Build + Search BANN and sklearn KDTree search results do not match at dim = {dim}")
#        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
#
    print(f"bann: {np.mean(bann_t)} vs sklearn: {np.mean(sk_t)}")
