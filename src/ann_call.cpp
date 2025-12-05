#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

#include <math.h>

namespace ann_namespace {
  #include "ANN.h"
}

extern "C" {
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
  void bann_search(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                   int *K, int *Indx, double *Eps, int *DivChoice)
  {
    using namespace ann_namespace;

    const int dim = *Dim;
    const int nData = *NData;
    const int nQuery = *NQuery;
    const int k = *K;
    const double eps = *Eps;
    const int divChoice = *DivChoice;

    ANNkd_tree *tree;
    ANNpointArray dataPts = annAllocPts(nData, dim);
    ANNpointArray queryPts = annAllocPts(nQuery, dim);
    ANNidxArray nnIdx = new ANNidx[k];
    ANNdistArray divs = new ANNdist[k];

    int ptr = 0;
    /* Read in data points.
     *  Data is input as a contiguous block, passed in row-major order.
     */
    for (int i = 0; i < nData; i++) {
      for (int j = 0; j < dim; j++) {
        dataPts[i][j] = Data[i * dim + j];
      }
    }
    tree = new ANNkd_tree(dataPts, nData, dim);
    /* Read in query points
     *  Query is input as a contiguous block, passed in row-major order.
    */
    for (int i = 0; i < nQuery; i++) {
      for (int j = 0; j < dim; j++) {
        queryPts[i][j] = Query[i * dim + j];
      }
    }

    /* For each query point, find the k nearest neighbors. 
     *   Store indices in Indx array.
    */
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
        std::cerr << "Directive: "<< divChoice << "\n";
        break;
    }
    annDeallocPts(dataPts);
    annDeallocPts(queryPts);
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
      using namespace ann_namespace;

      const int dim = *Dim;
      const int nP = *NP;
      const int nQ = *NQ;
      const double eps = *Eps;
      const int divChoice = *DivChoice;

      ANNkd_tree *tree;
      ANNpointArray dataPts = annAllocPts(nP, dim);
      ANNpointArray queryPts = annAllocPts(nQ, dim);

      ANNidxArray nnIdx = new ANNidx[1];
      ANNdistArray divs = new ANNdist[1];

      double hausdorff = 0.0;

      /* Read in Data pts, and build the kd_tree
       * */
      for (int i = 0; i < nP; i++) {
         for (int j = 0; j < dim; j++) {
            dataPts[i][j] = P[i * dim + j];
         }
      }
      /* Build kd-tree on P
       * */
      tree = new ANNkd_tree(dataPts, nP, dim);
      /* Read in Query points
       * */
      for (int i = 0; i < nQ; i++) {
         for (int j = 0; j < dim; j++) {
            queryPts[i][j] = Q[i * dim + j];
         }
      }
      /* Direction notes:
       * By default, the BH search builds the kd-tree on the first set (P), and then 
       * computes the nearest neighbour with the reversed computation direction from
       * the nearest neighbour search definition. Thus, we artificially reverse the 
       * order of computations in this switch statement.
      */
      switch (divChoice) {
         case 0: // (squared) Euclidean search
            for (int i = 0; i < nQ; i++) {
               tree->annhSearch(
                     div_component_eucl,
                     queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
      annDeallocPts(dataPts);
      annDeallocPts(queryPts);
      delete tree;
      delete [] nnIdx;
      delete [] divs;

      return hausdorff;
   }


  /* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
   * Timing functions 
   *  Repeat the functions above but with timings for each.
   * -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
  */
   std::chrono::time_point<std::chrono::system_clock> print_time(
         std::chrono::time_point<std::chrono::system_clock> start, 
         std::chrono::time_point<std::chrono::system_clock> end,
         const std::string& task)
   {
      std::chrono::duration<double> elapsed = end - start;
      std::cout << task << " Duration: " << elapsed.count() << "s" << std::endl;
      return std::chrono::system_clock::now();
   }

   void timed_search(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                   int *K, int *Indx, double *Eps, int *DivChoice)
   {
      using namespace ann_namespace;

    const int dim = *Dim;
    const int nData = *NData;
    const int nQuery = *NQuery;
    const int k = *K;
    const double eps = *Eps;
    const int divChoice = *DivChoice;

    ANNkd_tree *tree;
    ANNpointArray dataPts = annAllocPts(nData, dim);
    ANNpointArray queryPts = annAllocPts(nQuery, dim);
    ANNidxArray nnIdx = new ANNidx[k];
    ANNdistArray divs = new ANNdist[k];

    int ptr = 0;
    /* Read in data points.
     *  Data is input as a contiguous block, passed in row-major order.
     */
    std::chrono::time_point<std::chrono::system_clock> phase_1, phase_2;
    phase_1 = std::chrono::system_clock::now();
    for (int i = 0; i < nData; i++) {
      for (int j = 0; j < dim; j++) {
        dataPts[i][j] = Data[i * dim + j];
      }
    }
    phase_2 = std::chrono::system_clock::now();
    phase_1 = print_time(phase_1, phase_2, "Read data");
    
    tree = new ANNkd_tree(dataPts, nData, dim);
    phase_2 = print_time(phase_2, phase_1, "Build tree");
    /* Read in query points
     *  Query is input as a contiguous block, passed in row-major order.
    */
    for (int i = 0; i < nQuery; i++) {
      for (int j = 0; j < dim; j++) {
        queryPts[i][j] = Query[i * dim + j];
      }
    }
    phase_1 = print_time(phase_1, phase_2, "Read Query");

    /* For each query point, find the k nearest neighbors. 
     *   Store indices in Indx array.
    */
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
        std::cerr << "Directive: "<< divChoice << "\n";
        break;
    }
    phase_2 = print_time(phase_2, phase_1, "k_search");
    annDeallocPts(dataPts);
    annDeallocPts(queryPts);
    delete tree;
    delete [] nnIdx;
    delete [] divs;
  }
   double timed_haus(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
        double *Eps, int*DivChoice)
   {
      using namespace ann_namespace;

      const int dim = *Dim;
      const int nData = *NData;
      const int nQ = *NQuery;
      const double eps = *Eps;
      const int divChoice = *DivChoice;

      ANNkd_tree *tree;
      ANNpointArray dataPts = annAllocPts(nData, dim);
      ANNpointArray queryPts = annAllocPts(nData, dim);

      ANNidxArray nnIdx = new ANNidx[1];
      ANNdistArray divs = new ANNdist[1];

      double hausdorff = 0.0;

       std::chrono::time_point<std::chrono::system_clock> phase_1, phase_2;
       phase_1 = std::chrono::system_clock::now();
      /* Read in Data pts, and build the kd_tree
       * */
      for (int i = 0; i < nData; i++) {
         for (int j = 0; j < dim; j++) {
            dataPts[i][j] = Data[i * dim + j];
         }
      }
      phase_2 = std::chrono::system_clock::now();
      phase_1 = print_time(phase_1, phase_2, "Read data");
      tree = new ANNkd_tree(dataPts, nData, dim);
      phase_2 = print_time(phase_2, phase_1, "Build tree");
      /* Read in Query points
       * */
      for (int i = 0; i < nQ; i++) {
         for (int j = 0; j < dim; j++) {
            queryPts[i][j] = Query[i * dim + j];
         }
      }
      phase_1 = print_time(phase_1, phase_2, "read query");
      switch (divChoice) {
         case 0: // (squared) Euclidean search
            for (int i = 0; i < nQ; i++) {
               tree->annhSearch(
                     div_component_eucl,
                     queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
                  queryPts[i],
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
      phase_2 = print_time(phase_2, phase_1, "Haus search");
      annDeallocPts(dataPts);
      annDeallocPts(queryPts);
      delete tree;
      delete [] nnIdx;
      delete [] divs;
      return hausdorff;
   }
}
