#pragma once

#include <cstdlib>			// standard lib includes
#include <cmath>			// math includes
#include <cassert>
#include <string>
#include <functional>

/* ANN generates points in the box (-1, 1) ^ 2
 which is not great for KL, IS etc.
 so this function transforms these points
 so that they are in (0,1)^2.

 TODO: Not sure if they do () or [] or (].
*/
inline void adjust(double& p_i, double& q_i)
{
	p_i += 1;
	q_i += 1;
	p_i /= 2;
	q_i /= 2;
}

/* File for defining divergences. Note that the Kd-tree has only been proven to
 * be effective for `decomposable Bregman divergences.'
 *
 * For new divergence implementation, you only need to define *a dimensional
 * component* of any chosen decomposable Bregman divergence extend the required 
 * switch statements.
 *
 * The following programs call div_component as templates:
 *    kd_search.cpp
 *    kd_util.cpp
 *    kd_haus.cpp
 *    ANNx.h
 *
 * The following files should have their calls adjusted:
 *    
 */
 using divergence = std::function<double(const double, const double)>;
 
inline double div_component_eucl(const double p_i, const double q_i) {
	return (p_i - q_i) * (p_i - q_i);
}

inline double div_component_kl(const double p_i, const double q_i)
{
	assert(q_i > 0);

	if (p_i == 0 && q_i > 0)
		return q_i;

	return p_i * log(p_i) - p_i * log(q_i) - p_i + q_i;
}

inline double div_component_is(const double p_i, const double q_i)
{
	assert(q_i > 0);
	assert(p_i > 0);

	return p_i / q_i - log(p_i) + log(q_i) - 1;
}

inline double div_component_dkl(const double p_i, const double q_i) {
  return div_component_kl(q_i, p_i);
}

inline double div_component_dis(const double p_i, const double q_i) {
  return div_component_is(q_i, p_i);
}

//#define div_component div_component_dis
