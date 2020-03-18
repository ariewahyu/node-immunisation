""" Several utility functions """

import igraph as ig
import numpy as np
import itertools


def get_graph(name):
    """
    Gets a graph stored in the GML file format and returns the largest component

    Parameters
    ----------
    name: string
        Filename

    Returns
    --------
        Igraph object of file
    """

    disconn = ig.Graph.Read_GML(name)
    graph = disconn.components(mode=ig.WEAK).giant()
    return graph


def get_adj_np(graph, attribute=None):
    """
    Transform the igraph graph object into a 2D numpy array

    Parameters
    ----------
    graph: igraph Graph object
        Igraph graph object to convert
    attributes:  String, optional
        Attribute of the edges in the graph object with which to fill the adjacency matrix.
        Default simply returns unweighted matrix with 1s and 0s.

    Returns
    --------
        2D numpy array
    """
    adj = graph.get_adjacency(type=ig.GET_ADJACENCY_BOTH, attribute=attribute)
    return np.array(adj.data)


def get_graph_np(adj):
    """
    Transform the 2D numpy adjacency matrix into an igraph graph object

    Parameters
    ----------
    adj: 2D numpy array.
        Adjacency matrix to convert

    Returns
    --------
        igraph Graph object
    """

    return ig.Graph.Adjacency(adj.tolist(), mode=ig.ADJ_UNDIRECTED)


def get_max_eigen(adj):
    """
    Get the maximum eigenvalue and corresponding eigenvector of a symmetric matrix

    Parameters
    ----------
    adj: 2D numpy array.
        Matrix to perform eigendecomposition on

    Returns
    --------
        Returns tuple of (eigenvalue, eigenvector)
    """
    eigvals, eigvecs = np.linalg.eigh(adj)
    return np.abs(eigvals[-1]), np.abs(eigvecs[:, -1])


def get_max_eigenvalue(adj):
    """
    Get the maximum eigenvalue of a symmetric matrix

    Parameters
    ----------
    adj: 2D numpy array.
        Matrix to perform eigendecomposition on

    Returns
    --------
        Returns eigenvalue
    """

    eigvals = np.linalg.eigvalsh(adj)
    return np.abs(eigvals[-1])



def largest_degree_immunization(graph, k):
    """
    Returns the indices of the k nodes with the largest degrees from the graph

    Parameters
    ----------
    graph: igraph graph object.
        Input graph

    Returns
    --------
        1D numpy array:
            The inidices of the k nodes with the largest degrees
    """

    degrees = np.array(graph.degree())
    return degrees.argsort()[::-1][:k]


def shieldvalue(S, adj):
    """ Computes the Shield-value of an adjacency matrix and selected vertices S """


    eigvals, eigvecs = np.linalg.eig(adj)
    max_eig_ind = np.argmax(np.real(eigvals))

    lambd = np.real(eigvals[max_eig_ind])
    u = np.real(eigvecs[:, max_eig_ind])

    Sv_p = 0
    for i in S:
        Sv_p += 2*lambd*np.square(u[i])

    Sv_q = 0
    for i, j in itertools.combinations(S, 2):
        Sv_q += 2 * adj[i, j] * u[i] * u[j]

    return Sv_p - Sv_q
