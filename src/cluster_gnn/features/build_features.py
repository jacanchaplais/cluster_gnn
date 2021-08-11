import numpy as np
import vector

def _array_to_vec(array):
    """Takes input of (n,4)-d array of n particles 4-momenta, and
    returns an array of Lorentz 4-vectors.

    Note: assumes format x, y, z, e.
    """
    return vector.array(
        [tuple(pcl) for pcl in array],
        dtype=[("x", float), ("y", float), ("z", float), ("e", float)]
        )

def __get_array_size(array):
    if type(array) == vector.MomentumNumpy4D:
        size = array.size
    elif type(array) == np.ndarray:
        size = array.shape[0]
    else:
        raise TypeError('input 4 momenta array not ndarray or vector')
    return size

def _deltaR_cols(array):
    array = _array_to_vec(array)
    size = __get_array_size(array)
    # slide the particle lists over all pairs
    for shift in range(size): # 0th shift is trivial as both same
        yield array[shift].deltaR(array[shift:])

def deltaR_aff(array):
    """Returns a symmetric matrix of delta R vals from input 4-momentum
    array.
    """
    size = __get_array_size(array)
    aff = np.zeros((size, size), dtype=np.float64)
    dR_cols = _deltaR_cols(array)
    for idx, col in enumerate(dR_cols):
        aff[idx:, idx] = col
        aff[idx, idx:] = col
    return aff

def knn_adj(matrix, self_loop=False, k=8, weighted=False, row=True,
            dtype=np.bool_):
    """Produce a directed adjacency matrix with outward edges towards
    the k nearest neighbours, determined from the input affinity matrix.
    
    Keyword arguments:
    matrix -- 2d floating point numpy array: particle affinities
    self_loop -- bool: if False will remove self-edges
    k -- int: number of nearest neighbours in result
    weighted -- bool: if True edges weighted by affinity,
        if False edge is binary
    row -- bool: if True, outward edges given by rows, if False columns
    dtype -- numpy data type: type of the output array
        note: must be floating point if weighted is True
    """
    axis = 0 # calculate everything row-wise
    if self_loop == False:
        k = k + 1 # for when we get rid of self-neighbours
    knn_idxs = np.argpartition(matrix, kth=k, axis=axis)
    near = knn_idxs[:k]
    edge_weights = 1
    if weighted == True:
        if not isinstance(dtype(1), np.floating):
            raise ValueError(
                "Data type must be a numpy float for weighted output")
        edge_weights = np.take_along_axis(matrix, near, axis=axis)
    adj = np.zeros_like(matrix, dtype=dtype)
    np.put_along_axis(adj, near, edge_weights, axis=axis)
    if row == False:
        adj = adj.T
    if self_loop == False:
        np.fill_diagonal(adj, 0)
    return adj

def fc_adj(num_nodes, self_loop=False, dtype=np.bool_):
    """Create a fully connected adjacency matrix.
    """
    adj = np.ones((num_nodes, num_nodes), dtype=dtype)
    if self_loop == False:
        np.fill_diagonal(adj, 0)
    return adj
