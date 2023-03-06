
import os
import sys
import csv
import numpy as np

from sklearn.neighbors import NearestNeighbors


#chroms, positions = read_msms(filename="SimsOuts/5/NE_1.txt", NCHROMS=40)

#chroms = order_data(im_matrix=chroms, sort="freq", positions=positions, winsize=20) 


def read_msms(filename, NCHROMS):
    """
    Reads msms file to an haplotype matrix

    Parameters:
        filename: full path and name of the .txt MSMS file
        NCHROMS: number of samples(haploid individuals, or chromosoms
    Output:
        Returns an haplotype array, and an array containing positions
    """
    file = open(filename).readlines()
    if len(file) == 0:
        raise Exception('The file {} is empty'.format(filename.split('/')[-1]))
    # look for the // character in the file
    pointer = file.index('//\n') + 3

    # Get positions
    pos = file[pointer - 1].split()
    del pos[0]
    pos = np.array(pos, dtype='float')

    # Get the number of genomic positions(determined be the number or pointers)
    n_columns = len(list(file[pointer])) - 1
    # Intialize the empty croms matrix: of type: object
    croms = np.empty((NCHROMS, n_columns), dtype=np.object)
    # Fill the matrix with the simulated data
    for j in range(NCHROMS):
        f = list(file[pointer + j])
        del f[-1]
        F = np.array(f)
        croms[j, :] = F
    croms = croms.astype(int)
    return croms, pos


def sort_freq(im_matrix):
    """
    This function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted
    by genetic similarity.

    Parameters:
        im_matrix: Array containing sequence data
    Returns:
        Sorted array containing sequence data
    """
    # u: Sorted Unique arrays
    # index: Index of 'im_matrix' that corresponds to each unique array
    # count: The number of instances each unique array appears in the 'im_matrix'
    u, index, count = np.unique(im_matrix, return_index=True, return_counts=True, axis=0)
    # b: Intitialised matrix the size of the original im_matrix[where new sorted data will be stored]
    b = np.zeros((np.size(im_matrix, 0), np.size(im_matrix, 1)), dtype=int)
    # c: Frequency table of unique arrays and the number of times they appear in the original 'im_matrix'
    c = np.stack((index, count), axis=-1)
    # The next line sorts the frequency table based mergesort algorithm
    c = c[c[:, 1].argsort(kind='mergesort')]
    pointer = np.size(im_matrix, 0) - 1
    for j in range(np.size(c, 0)):
        for conta in range(c[j, 1]):
            b[pointer, :] = im_matrix[c[j, 0]]
            pointer -= 1
    return b

def sort_min_diff(im_matrix):
    """
    This function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted
    by genetic similarity. this problem is NP, so here we use a nearest neighbors approx.  it's not perfect,
    but it's fast and generally performs ok.
    Implemented from https://github.com/flag0010/pop_gen_cnn/blob/master/sort.min.diff.py#L1

    Parameters:
        im_matrix: haplotype matrix (np array)

    Returns:
        Sorted numpy array
    """
    mb = NearestNeighbors(len(im_matrix), metric='manhattan').fit(im_matrix)
    v = mb.kneighbors(im_matrix)
    smallest = np.argmin(v[0].sum(axis=1))
    return im_matrix[v[1][smallest]]


def order_data(im_matrix, sort, positions, winsize):
    """
    Sorts haplotype matrix

    Parameters:
        im_matrix: input haplotype matrix
        sort: sorting method. either
            gen_sim: based on genetic similarity, or
            freq: based on frequency
        positions: positions
        N: nr of base pairs
        winsize: half of the window size for croppin

    Returns:
        sorted haplotype matrix
    """
    if len(positions)<(winsize*2):
        raise ValueError("not enough snps")
    midpoint = np.argmin(abs(positions-0.5))
    if sort == "gen_sim":
        croms = sort_min_diff(im_matrix[:,(midpoint-winsize):(midpoint+winsize)])
    elif sort == "freq":
        croms = sort_freq(im_matrix[:,(midpoint-winsize):(midpoint+winsize)])
    else:
        croms = im_matrix[:,(midpoint-winsize):(midpoint+winsize)]
    return croms



