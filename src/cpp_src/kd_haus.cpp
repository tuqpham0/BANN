
#include "kd_haus.h"

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
////----------------------------------------------------------------------
//
//int				ANNkdDim;				// dimension of space
//ANNpoint		   ANNkdQ;					// query point
//double			ANNkdMaxErr;			// max tolerable squared error
//ANNpointArray	ANNkdPts;				// the points
//ANNmin_k		   *ANNkdPointMK;			// set of k closest points

//----------------------------------------------------------------------
//	annhSearch - Search for Bregman--Hausdorff divergence of two sets
//----------------------------------------------------------------------
void ANNkd_tree::annhSearch(
      divergence     div_component,
      ANNpoint       q,
      ANNidxArray    nn_idx,
      ANNdistArray   dd,
      double         eps,
      double         haus)
{
   ANNkdDim = dim;
   ANNkdQ = q;
   ANNkdPts = pts;
   ANNptsVisited = 0;
   
   ANNkdMaxErr = 1.0 + eps;

   ANNkdPointMK = new ANNmin_k(1);

   root->ann_haus(annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim, div_component), div_component, haus);
   
   // adjusted since we only need 1 the nearest neighbor for Hausdorff
   dd[0] = ANNkdPointMK->ith_smallest_key(0);
   nn_idx[0] = ANNkdPointMK->ith_smallest_info(0);
   //
   
   delete ANNkdPointMK;
}

//----------------------------------------------------------------------
// kd_split::ann_haus - query a splitting node	
//----------------------------------------------------------------------
void ANNkd_split::ann_haus(ANNdist box_dist, divergence div_component, double haus)
{
   ANNdist min_dist;

   min_dist = ANNkdPointMK->max_key();
   if (min_dist < haus) {
      return;
   }
	
	if (ANNmaxPtsVisited != 0 && ANNptsVisited > ANNmaxPtsVisited) return;

   ANNcoord cut_diff = ANNkdQ[cut_dim] - cut_val;

   if (cut_diff < 0) {
      child[ANN_LO]->ann_haus(box_dist, div_component, haus);

      auto new_dist = box_dist + div_component(ANNkdQ[cut_dim], cut_val);
      
      auto box_diff = cd_bnds[ANN_LO] - ANNkdQ[cut_dim];

      if (box_diff > 0)
         new_dist -= div_component(ANNkdQ[cut_dim], cd_bnds[ANN_LO]);

      if (box_dist * ANNkdMaxErr < ANNkdPointMK->max_key())
         child[ANN_HI]->ann_haus(new_dist, div_component, haus);
   }
   else {
      child[ANN_HI]->ann_haus(box_dist, div_component, haus);

		auto new_dist = box_dist + div_component(ANNkdQ[cut_dim], cut_val);

		auto box_diff = ANNkdQ[cut_dim] - cd_bnds[ANN_HI];
      
      if (box_diff > 0)
         new_dist -= div_component(ANNkdQ[cut_dim], cd_bnds[ANN_HI]);

      if (box_dist * ANNkdMaxErr < ANNkdPointMK->max_key())
         child[ANN_LO]->ann_haus(new_dist, div_component, haus);
   }
}

//----------------------------------------------------------------------
// kd_leaf::ann_haus - Search points in a leaf. If the divergence is too low,
//    then we can abort early.
//----------------------------------------------------------------------
void ANNkd_leaf::ann_haus(ANNdist box_dist, divergence div_component, double haus)
{
   ANNdist dist;
   ANNcoord* pp;
   ANNcoord* qq;
   ANNdist min_dist;
   ANNcoord t;
   int d;

   min_dist = ANNkdPointMK->max_key();

   for (int i = 0; i < n_pts; i++) {
      pp = ANNkdPts[bkt[i]];
      qq = ANNkdQ;
      dist = 0;

      for (d = 0; d < ANNkdDim; d++) {
         dist += div_component(*qq++, *pp++);

         if (dist > min_dist)
            break;
      }
      if (d >= ANNkdDim && 
            (ANN_ALLOW_SELF_MATCH || dist != 0)) {
            ANNkdPointMK->insert(dist, bkt[i]);
            min_dist = ANNkdPointMK->max_key();
      }
      // If the current candidate is less than Hausdorff, the nearest neighbor will be
      // too, so we can terminate the search early
      if (min_dist < haus)
         return;
   }
}
