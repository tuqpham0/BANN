import numpy
cimport numpy

cdef extern from "ann_call.cpp":
    void bann_search(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                     int *K, int *Indx, double *Eps, int *DivChoice)
    void timed_search(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                     int *K, int *Indx, double *Eps, int *DivChoice)
    double bann_haus(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                     double *Eps, int *DivChoice)
    double timed_haus(double *Data, int *NData, double *Query, int *NQuery, int *Dim,
                     double *Eps, int *DivChoice)

def k_search(
    numpy.ndarray[double, ndim=2] data,
    numpy.ndarray[double, ndim=2] query,
    int k = 1, double eps = 0, str div = 'kl') -> numpy.ndarray:
    """
    Bregman Nearest Neighbour search
    Uses a kd-tree to find the $k$-nearest neighbours for each point in input query set from
    the input data set based on the specified decomposable Bregman divergence.

    This function is a wrapper for the ANN library with decomposable Bregman divergences
    support. It utilizes the Approximate Nearest Neighbour (ANN) C++ library developed by
    Mount and Arya, with extensions for Bregman divergences and the Bregman--Hausdorff
    divergence made by Wagner and Pham.

    More information about the underlying ANN library can be found at:
       http://www.cs.umd.edu/~mount/ANN/

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (n_points, dim) representing the data set.
    query : numpy.ndarray
        A 2D numpy array of shape (m_points, dim) representing the query points.
    k : int, optional
        The number of nearest neighbors to search for. Default is 1.
    eps : float, optional
        The error tolerance for the search. Default is 0.0.
    divergence : str, optional
        The type of Bregman divergence to use. Default is 'kl'. 
        As Bregman divergences are asymmetric, options are:
           'se'  - Squared Euclidean
           'kl'  - Kullback-Leibler
           'dkl' - Dual Kullback-Leibler
           'is'  - Itakura-Saito
           'dis' - Dual Itakura-Saito
    
    Returns
    -------
    indices : numpy.ndarray
        A 2D numpy array of shape (m_points, k) containing the indices of the k-nearest neighbors
        in the data set for each query point.
    """
    # Parse inputs and check validity at Python level
    ndata, dim = data.shape[0], data.shape[1]
    nquery, qdim = query.shape[0], query.shape[1]
    if dim != qdim:
        raise ValueError("Data points and query points must lie in the same dimension.")
    if k > ndata or k <= 0:
        raise ValueError("Must search for at least 1 nearest neighbour and less neighbours than data.")

    div_map = {
        'se': 0,
        'kl': 1,
        'dkl': 2,
        'is': 3,
        'dis': 4
    }
    if not isinstance(div, str):
        raise TypeError("Divergence choice must be a string.")
    try:
        DivChoice = div_map[div.lower()]
    except KeyError:
        raise ValueError(f"Unknown divergence choice '{div}'. Supported choices are: {list(div_map.keys())}.")

    # Convert to C-types
    cdef int ND = ndata
    cdef int NQ = nquery
    cdef int D = dim
    cdef int K = k
    cdef double Eps = eps
    cdef int divChoice = DivChoice

    cdef numpy.ndarray[double, ndim=1] data_c = numpy.ascontiguousarray(data.ravel(), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=1] query_c = numpy.ascontiguousarray(query.ravel(), dtype=numpy.double)
    cdef double *data_ptr = &data_c[0] if data_c.size else NULL
    cdef double *query_ptr = &query_c[0] if query_c.size else NULL
    # Prepare output array
    cdef numpy.ndarray[int, ndim=1] nn_index = numpy.empty(NQ * K, dtype=numpy.intc)

    # Call to C++ (Release Global Interpreter Lock since ANN is pure C++)
    bann_search(data_ptr, &ND, query_ptr, &NQ, &D, &K, &nn_index[0], &Eps, &divChoice)

    return nn_index.reshape((NQ, K))

def bhaus(
    numpy.ndarray[double, ndim=2] setp,
    numpy.ndarray[double, ndim=2] setq,
    double eps = 0, str div = 'kl') -> double:
    """
    (Approximate) Bregman--Hausdorff divergence search:
    Uses a kd-tree to find the Bregman--Hausdorff divergence from a set of vectors $A$
    to another set of vectors $B$.
    This function is a wrapper for the Bregman--Hausdorff divergence shell algorithm.
    It is an extension developed by Wagner and Pham of the the Approximate Nearest
    Neighbour (ANN) C++ library which was developed by Mount and Arya.

    More information about the Bregman--Hausdorff divergence can be found at:
      https://arxiv.org/abs/2504.07322

    Note that the direction of computation is reversed when computing the
    Bregman--Hausdorff divergence.

    Parameters
    ----------
    data: numpy.ndarray
        A 2D numpy array of shape (n_points, dim) representing the data set.
    query : numpy.ndarray
        A 2D numpy array of shape (m_points, dim) representing the query points.
    eps : float, optional
        The error tolerance for the search to return a (1+eps) approximation. Default is 0.0.
    divergence : str, optional
        The type of Bregman divergence to use. Default is 'kl'.
        As Bregman divergences are asymmetric, the opposite directions are separate options.
        Currently supported options are:
            'se'  - Squared Euclidean
            'kl'  - Kullback-Leibler
            'dkl' - Dual Kullback-Leibler
            'is'  - Itakura-Saito
            'dis' - Dual Itakura-Saito

    Returns
    -------
    haus : double
        The Bregman--Hausdorff divergence from query $\to$ data.
    """
    # Parse inputs and check validity at Python level
    np, dim = setp.shape[0], setp.shape[1]
    nq, qdim = setq.shape[0], setq.shape[1]
    if dim != qdim:
        raise ValueError("P and Q must have the same dimension.")

    div_map = {
        'se': 0,
        'kl': 1,
        'dkl': 2,
        'is': 3,
        'dis': 4
    }
    if not isinstance(div, str):
        raise ValueError("Divergence choice must be a string.")
    try:
        DivChoice = div_map[div.lower()]
    except KeyError:
        raise ValueError(f"Unknown divergence choice '{div}'. Supported choices are: {list(div_map.keys())}.")

    # Convert to C-types
    cdef int ND = np
    cdef int NQ = nq
    cdef int D = dim
    cdef double Eps = eps
    cdef int divChoice = DivChoice

    cdef numpy.ndarray[double, ndim=1] data_c = numpy.ascontiguousarray(setp.ravel(), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=1] query_c = numpy.ascontiguousarray(setq.ravel(), dtype=numpy.double)
    cdef double *data_ptr = &data_c[0] if data_c.size else NULL
    cdef double *query_ptr = &query_c[0] if query_c.size else NULL

    haus_div = bann_haus( data_ptr, &ND, query_ptr, &NQ, &D, &Eps, &divChoice )

    return haus_div


#--------------------------------------------------------------------------------------------------
# Functions for C++ timings
#--------------------------------------------------------------------------------------------------
def __timed_k_search(
    numpy.ndarray[double, ndim=2] data,
    numpy.ndarray[double, ndim=2] query,
    int k = 1, double eps = 0, str div = 'kl') -> numpy.ndarray:
    """
        k_search but with times for each operations for testing purposes.
    """
    # Parse inputs and check validity at Python level
    ndata, dim = data.shape[0], data.shape[1]
    nquery, qdim = query.shape[0], query.shape[1]
    if dim != qdim:
        raise ValueError("Data points and query points must lie in the same dimension.")
    if k > ndata or k <= 0:
        raise ValueError("Must search for at least 1 nearest neighbour and less neighbours than data.")

    div_map = {
        'se': 0,
        'kl': 1,
        'dkl': 2,
        'is': 3,
        'dis': 4
    }
    if not isinstance(div, str):
        raise ValueError("Divergence choice must be a string.")
    try:
        DivChoice = div_map[div.lower()]
    except KeyError:
        raise ValueError(f"Unknown divergence choice '{div}'. Supported choices are: {list(div_map.keys())}.")

    # Convert to C-types
    cdef int ND = ndata
    cdef int NQ = nquery
    cdef int D = dim
    cdef int K = k
    cdef double Eps = eps
    cdef int divChoice = DivChoice

    cdef numpy.ndarray[double, ndim=1] data_c = numpy.ascontiguousarray(data.ravel(), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=1] query_c = numpy.ascontiguousarray(query.ravel(), dtype=numpy.double)
    cdef double *data_ptr = &data_c[0] if data_c.size else NULL
    cdef double *query_ptr = &query_c[0] if query_c.size else NULL
    # Prepare output array
    cdef numpy.ndarray[int, ndim=1] nn_index = numpy.empty(NQ * K, dtype=numpy.intc)

    # Call to C++ (Release Global Interpreter Lock since ANN is pure C++)
    timed_search(data_ptr, &ND, query_ptr, &NQ, &D, &K, &nn_index[0], &Eps, &divChoice)

    return nn_index.reshape((NQ, K))

def __timed_bhaus(
    numpy.ndarray[double, ndim=2] data,
    numpy.ndarray[double, ndim=2] query,
    double eps = 0, str div = 'kl') -> double:
    # Parse inputs and check validity at Python level
    ndata, dim = data.shape[0], data.shape[1]
    nquery, qdim = query.shape[0], query.shape[1]
    if dim != qdim:
        raise ValueError("Data points and query points must lie in the same dimension.")

    div_map = {
        'se': 0,
        'kl': 1,
        'dkl': 2,
        'is': 3,
        'dis': 4
    }
    if not isinstance(div, str):
        raise ValueError("Divergence choice must be a string.")
    try:
        DivChoice = div_map[div.lower()]
    except KeyError:
        raise ValueError(f"Unknown divergence choice '{div}'. Supported choices are: {list(div_map.keys())}.")

    # Convert to C-types
    cdef int ND = ndata
    cdef int NQ = nquery
    cdef int D = dim
    cdef double Eps = eps
    cdef int divChoice = DivChoice

    cdef numpy.ndarray[double, ndim=1] data_c = numpy.ascontiguousarray(data.ravel(), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=1] query_c = numpy.ascontiguousarray(query.ravel(), dtype=numpy.double)
    cdef double *data_ptr = &data_c[0] if data_c.size else NULL
    cdef double *query_ptr = &query_c[0] if query_c.size else NULL

    haus = timed_haus(data_ptr, &ND, query_ptr, &NQ, &D, &Eps, &divChoice)
    return haus
