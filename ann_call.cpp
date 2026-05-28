#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

#include <math.h>

#include "ANN.h"

namespace ann_namespace {
  extern "C++" {
    void bann_search(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                    int *K, int *Indx, double *Eps, int *DivChoice)
    {
    /* ANN search wrapper 
      * Performs k-nearest neighbor search using specified divergence.
      *  
      *  Inputs:
      *    Data     - pointer to data points (row-major order)
      *    NData    - number of data points
      *    Query    - pointer to query points (row-major order)
      *    NQuery   - number of query points
      *    Dim      - dimension of points
      *    K        - number of nearest neighbors to find
      *    Eps      - approximation factor
      *    DivChoice- divergence choice (0: Eucl, 1: KL, 2: DKL, 3: IS, 4: DIS)
      *  
      *  Output: None
      *    Stores array of indices of k nearest neighbours of each query point in Indx
      *    (row-major order)
    */
      const int dim = *Dim;
      const int nData = *NData;
      const int nQuery = *NQuery;
      const int k = *K;
      const double eps = *Eps;
      const int divChoice = *DivChoice;

      ANNkd_tree *tree;
      ANNidxArray nnIdx = new ANNidx[k];
      ANNdistArray divs = new ANNdist[k];
      
      // Create array of pointers to rows in the flat data array
      ANNpointArray dataPts = new ANNpoint[nData];
      // ANNpointArray queryPts = new ANNpoint[nQuery];
      
      // Convert numpy arrays to ANNpointArray format
      double* dataPtr = Data;
      for (int i = 0; i < nData; i++) {
          dataPts[i] = dataPtr;
          dataPtr += dim;
      }
      
      // double* queryPtr = Query;
      // for (int i = 0; i < nQuery; i++) {
      //     queryPts[i] = queryPtr;
      //     queryPtr += dim;
      // }

      int ptr = 0;
      tree = new ANNkd_tree(dataPts, nData, dim);

      /* For each query point, find the k nearest neighbors. 
      *   Store indices in Indx array.
      */
      switch (divChoice) {
        case 0: // Euclidean search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_eucl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 1: // KL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_kl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 2: // DKL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dkl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 3: // IS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_is,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 4: // DIS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dis,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        default:
          break;
      }
      
      // Memory cleanup
      delete [] dataPts;
      // delete [] queryPts;
      delete tree;
      delete [] nnIdx;
      delete [] divs;
    }

    /* ANN hausdorff search wrapper 
      * Performs approximate hausdorff distance computation using specified divergence.
      *  
      *  Inputs:
      *    Data     - pointer to data points (row-major order)
      *    NData    - number of data points
      *    Query    - pointer to query points (row-major order)
      *    NQuery   - number of query points
      *    Dim      - dimension of points
      *    Eps      - approximation factor
      *    DivChoice- divergence choice (0: Eucl, 1: KL, 2: DKL, 3: IS, 4: DIS)
      *  
      *  Output:
      *    (1+epsilon) hausdorff divergence
    */
    double bann_haus(double *P, int *NP, double *Q, int *NQ, int *Dim,
          double *Eps, int *DivChoice)
    {
        const int dim = *Dim;
        const int nP = *NP;
        const int nQ = *NQ;
        const double eps = *Eps;
        const int divChoice = *DivChoice;

        ANNkd_tree *tree;
        ANNidxArray nnIdx = new ANNidx[1];
        ANNdistArray divs = new ANNdist[1];
        double hausdorff = 0.0;

        // Convert numpy arrays to ANNpointArray format
        ANNpointArray PPts = new ANNpoint[nP];
        double* PPtr = P;
        for (int i = 0; i < nP; i++) {
            PPts[i] = PPtr;
            PPtr += dim;
        }

        tree = new ANNkd_tree(PPts, nP, dim);
        
        /* Direction notes:
          * Note that H_{D_F}(P\|Q) is not symmetric, and the order of computations matters.
          *     H_{D_F}(P\|Q) = max_{p in P} min_{q in Q} D_F(p\|q)
          *                   = Primal thickening of Q to contain P
          * To compute this, we build a kd-tree on the points in P and query with points in Q. Specifically,
          * For each q in Q, we have to compute with the divergences with direction of computation reversed.
          *   For example, for KL, we compute D_{KL}(q||p) instead of D_{KL}(p||q) as with the default Knn-search above.
          * For Squared Euclidean, this doesn't matter. For every other case, we use the dual divergence to the requested.
        */
        switch (divChoice) {
          case 0: // (squared) Euclidean search
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                      div_component_eucl,
                      Q + (i * dim),
                      // queryPts[i],
                      nnIdx,
                      divs,
                      eps,
                      hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 1: // H_{KL}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_dkl,
                    Q + (i * dim),
                    // queryPts[i],
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 2: // H'_{KL}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_kl,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 3: // H_{IS}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_dis,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                  if (hausdorff < divs[0]) {
                    hausdorff = divs[0];
                  }
              }
              break;
          case 4: // H'_{IS}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_is,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          default:
              std::cerr << "Directive: " << divChoice << "\n";
              break;
        }
        // Memory cleanup
        delete [] PPts;
        delete tree;
        delete [] nnIdx;
        delete [] divs;

        return hausdorff;
    }


    /* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
      * Timing functions 
      *  Repeat the functions above but with timings for each.
      *     These are for development purposes, and are not intended to be used in production code.
      *      Each major component of the algorithm is measured and reported as a string during the process.
      *      The timings are not returned to the caller, but are printed to standard output.
      * -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= */
    std::chrono::time_point<std::chrono::system_clock> time_pt() {
        return std::chrono::system_clock::now();
    }
    void print_time(
          std::chrono::time_point<std::chrono::system_clock> start, 
          std::chrono::time_point<std::chrono::system_clock> end,
          const std::string& task)
    {
        std::chrono::duration<double> elapsed = end - start;
        std::cout << task << " Duration: " << elapsed.count() << "s" << std::endl;
    }

    void streamed_query(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                    int *K, int *Indx, double *Eps, int *DivChoice)
    {
    /* ANN search wrapper 
      * Performs k-nearest neighbor search using specified divergence. Streams query from numpy array as the search occurs.
      *  
      *  Inputs:
      *    Data     - pointer to data points (row-major order)
      *    NData    - number of data points
      *    Query    - pointer to query points (row-major order)
      *    NQuery   - number of query points
      *    Dim      - dimension of points
      *    K        - number of nearest neighbors to find
      *    Eps      - approximation factor
      *    DivChoice- divergence choice (0: Eucl, 1: KL, 2: DKL, 3: IS, 4: DIS)
      *  
      *  Output: None
      *    Stores array of indices of k nearest neighbours of each query point in Indx
      *    (row-major order)
    */
      const int dim = *Dim;
      const int nData = *NData;
      const int nQuery = *NQuery;
      const int k = *K;
      const double eps = *Eps;
      const int divChoice = *DivChoice;
      std::chrono::time_point<std::chrono::system_clock> start, end;

      ANNkd_tree *tree;
      ANNidxArray nnIdx = new ANNidx[k];
      ANNdistArray divs = new ANNdist[k];
      
      // Create array of pointers to rows in the flat data array
      std::cout << "Creating ANNpointArray for data" << std::endl;
      start = time_pt();
      ANNpointArray dataPts = new ANNpoint[nData];
      
      // Convert numpy arrays to ANNpointArray format
      double* dataPtr = Data;
      for (int i = 0; i < nData; i++) {
          dataPts[i] = dataPtr;
          dataPtr += dim;
      }
      end = time_pt();
      print_time(start, end, "Create ANNpointArray for data");
      
      // ANNpointArray queryPts = new ANNpoint[nQuery];
      // double* queryPtr = Query;
      // for (int i = 0; i < nQuery; i++) {
      //     queryPts[i] = queryPtr;
      //     queryPtr += dim;
      // }

      int ptr = 0;
      start = time_pt();
      tree = new ANNkd_tree(dataPts, nData, dim);
      end = time_pt();
      print_time(start, end, "Building ANNkd_tree");

      start = time_pt();
      /* For each query point, find the k nearest neighbors. 
      *   Store indices in Indx array.
      */
      switch (divChoice) {
        case 0: // Euclidean search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_eucl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 1: // KL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_kl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 2: // DKL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dkl,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 3: // IS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_is,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 4: // DIS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dis,
              // queryPts[i],
              Query + (i * dim),
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        default:
          break;
      }
      end = time_pt();
      print_time(start, end, "ANN search");
      
      // Memory cleanup
      delete [] dataPts;
      // delete [] queryPts;
      delete tree;
      delete [] nnIdx;
      delete [] divs;
    }
    
    void preread_query(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                    int *K, int *Indx, double *Eps, int *DivChoice)
    {
    /* ANN search wrapper 
      * Performs k-nearest neighbor search using specified divergence. Reads Query into memory before the search.
      *  
      *  Inputs:
      *    Data     - pointer to data points (row-major order)
      *    NData    - number of data points
      *    Query    - pointer to query points (row-major order)
      *    NQuery   - number of query points
      *    Dim      - dimension of points
      *    K        - number of nearest neighbors to find
      *    Eps      - approximation factor
      *    DivChoice- divergence choice (0: Eucl, 1: KL, 2: DKL, 3: IS, 4: DIS)
      *  
      *  Output: None
      *    Stores array of indices of k nearest neighbours of each query point in Indx
      *    (row-major order)
    */
      const int dim = *Dim;
      const int nData = *NData;
      const int nQuery = *NQuery;
      const int k = *K;
      const double eps = *Eps;
      const int divChoice = *DivChoice;
      std::chrono::time_point<std::chrono::system_clock> start, end;

      ANNkd_tree *tree;
      ANNidxArray nnIdx = new ANNidx[k];
      ANNdistArray divs = new ANNdist[k];
      
      // Create array of pointers to rows in the flat data array
      std::cout << "Creating ANNpointArray for data" << std::endl;
      start = time_pt();
      ANNpointArray dataPts = new ANNpoint[nData];
      
      // Convert numpy arrays to ANNpointArray format
      double* dataPtr = Data;
      for (int i = 0; i < nData; i++) {
          dataPts[i] = dataPtr;
          dataPtr += dim;
      }
      end = time_pt();
      print_time(start, end, "Create ANNpointArray for data");
      
      start = time_pt();
      ANNpointArray queryPts = new ANNpoint[nQuery];
      double* queryPtr = Query;
      for (int i = 0; i < nQuery; i++) {
          queryPts[i] = queryPtr;
          queryPtr += dim;
      }
      end = time_pt();
      print_time(start, end, "Create ANNpointArray for query");

      int ptr = 0;
      start = time_pt();
      tree = new ANNkd_tree(dataPts, nData, dim);
      end = time_pt();
      print_time(start, end, "Building ANNkd_tree");

      /* For each query point, find the k nearest neighbors. 
      *   Store indices in Indx array.
      */
      start = time_pt();
      switch (divChoice) {
        case 0: // Euclidean search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_eucl,
              queryPts[i],
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 1: // KL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_kl,
              queryPts[i],
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 2: // DKL search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dkl,
              queryPts[i],
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 3: // IS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_is,
              queryPts[i],
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        case 4: // DIS search
          for (int i = 0; i < nQuery; i++) {
            tree->annkSearch(
              div_component_dis,
              queryPts[i],
              k,
              nnIdx,
              divs,
              eps);
            for (int j = 0; j < k; j++) {
              Indx[ptr++] = nnIdx[j];
            }
          }
          break;
        default:
          break;
      }
      end = time_pt();
      print_time(start, end, "ANN search");
      
      start = time_pt();
      // Memory cleanup
      delete [] dataPts;
      delete [] queryPts;
      delete tree;
      delete [] nnIdx;
      delete [] divs;
      end = time_pt();
      print_time(start, end, "Memory cleanup");
    }

    double timed_haus(double *P, int *NP, double *Q, int *NQ, int *Dim,
          double *Eps, int*DivChoice)
    {
        const int dim = *Dim;
        const int nP = *NP;
        const int nQ = *NQ;
        const double eps = *Eps;
        const int divChoice = *DivChoice;
      std::chrono::time_point<std::chrono::system_clock> start, end;

        ANNkd_tree *tree;
        ANNidxArray nnIdx = new ANNidx[1];
        ANNdistArray divs = new ANNdist[1];
        double hausdorff = 0.0;

        // Convert numpy arrays to ANNpointArray format
        start = time_pt();
        ANNpointArray PPts = new ANNpoint[nP];
        double* PPtr = P;
        for (int i = 0; i < nP; i++) {
            PPts[i] = PPtr;
            PPtr += dim;
        }
        end = time_pt();
        print_time(start, end, "Create ANNpointArray");

        start = time_pt();
        tree = new ANNkd_tree(PPts, nP, dim);
        end = time_pt();
        print_time(start, end, "Building ANNkd_tree");

        /* Direction notes:
          * Note that H_{D_F}(P\|Q) is not symmetric, and the order of computations matters.
          *     H_{D_F}(P\|Q) = max_{p in P} min_{q in Q} D_F(p\|q)
          *                   = Primal thickening of Q to contain P
          * To compute this, we build a kd-tree on the points in P and query with points in Q. Specifically,
          * For each q in Q, we have to compute with the divergences with direction of computation reversed.
          *   For example, for KL, we compute D_{KL}(q||p) instead of D_{KL}(p||q) as with the default Knn-search above.
          * For Squared Euclidean, this doesn't matter. For every other case, we use the dual divergence to the requested.
        */
        start = time_pt();
        switch (divChoice) {
          case 0: // (squared) Euclidean search
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                      div_component_eucl,
                      Q + (i * dim),
                      // queryPts[i],
                      nnIdx,
                      divs,
                      eps,
                      hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 1: // H_{KL}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_dkl,
                    Q + (i * dim),
                    // queryPts[i],
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 2: // H'_{KL}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_kl,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          case 3: // H_{IS}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_dis,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                  if (hausdorff < divs[0]) {
                    hausdorff = divs[0];
                  }
              }
              break;
          case 4: // H'_{IS}(P||Q)
              for (int i = 0; i < nQ; i++) {
                tree->annhSearch(
                    div_component_is,
                    Q + (i * dim),
                    nnIdx,
                    divs,
                    eps,
                    hausdorff);
                if (hausdorff < divs[0]) {
                  hausdorff = divs[0];
                }
              }
              break;
          default:
              std::cerr << "Directive: " << divChoice << "\n";
              break;
        }
        end = time_pt();
        print_time(start, end, "Bregman--Hausdorff search");
        // Memory cleanup
        delete [] PPts;
        delete tree;
        delete [] nnIdx;
        delete [] divs;

        return hausdorff;
    }
  } // close extern "C++"
} // close ann_namespace