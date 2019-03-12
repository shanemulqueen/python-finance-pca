import numpy
import pandas




def pca(data, cov_corr=True, eig_var=0.90, eig_num=None):
    """
    Calculates the Principal Component Analysis of a data matrix

    Args:
    data -- Assumes a pandas DataFrame. Can either be raw data or
           a covariance / correlation matrix
    cov_corr -- A bool set to True if the matrix data is a covariance
               or correlation matrix and False if data is a normal
               matrix of data. Defaults to True
    eig_var -- A maximum value of the cumulative variance to keep.
              Defaults to 90% or 0.90
    eig_num -- Instead of setting the cumulative variance maximum, can
              set eig_num to an integer value and explicitly return
              eig_num eigenvalues and eigenvectors. Defaults to None

    Returns:
    eigenvalues -- A eig_num X 1 numpy vector of eigenvalues
    eigenvectors -- A n X eig_num numpy array of eigenvectors
    """

    u,s,vh = numpy.linalg.svd(data)

    if cov_corr:
        eigenvalues = s
    else:
        eigenvalues = s*s

    if eig_num is None:
        cumulative_variance = (eigenvalues / eigenvalues.sum()).cumsum()
        for i,val in enumerate(cumulative_variance):
            if val > eig_var:
                eig_num = i
                break

    eigenvalues = eigenvalues[:eig_num].reshape(eig_num, 1)
    eigenvectors = vh.T[:,:eig_num]

    return eigenvalues, eigenvectors




def pca_window(data, window_length=60, eig_num=4, corr=True):
    """
    Runs PCA on a rolling window basis on the supplied data matrix

    Args:
    data -- Assumes a pandas dataframe of raw data
    window_length -- The size of each rolling window.
    eig_num -- Integer that determines the number of eigenvectors to keep in
              all windows
    corr - A bool for determining whether to calculate PCA over the
           covariance matrix or the correlation matrix

    Returns:
    eig_list -- A list that contains the eigenvalues, eigenvectors and
               timestamps of the beginning and ending observation in the
               window. Each entry in eig_list is a dictionary with the
               following keys:
               values - The eig_num x 1 vector of eigenvalues
               vectors - the n x eig_num matrix of eigenvectors
               begin - The timestamp for the first observation in the window
               end - The timestamp for the last observation in the window
    """
    t, n = data.shape

    eig_list = list()
    if corr:
        for i in xrange(window_length, t - window_length):
            temp_u, temp_s, temp_vh = numpy.linalg.svd(data[i:window_length+i].corr())
            eig_list.append({'values': temp_s[:eig_num].reshape(eig_num,1),
                             'vectors': temp_vh.T[:,:eig_num],
                             'begin': data.ix[i].name,
                             'end': data.ix[window_length+i].name})
    else:
        for i in xrange(window_length, t - window_length):
            temp_u, temp_s, temp_vh = numpy.linalg.svd(data[i:window_length+i].cov())
            eig_list.append({'values': (temp_s*temp_s)[:eig_num].reshape(eig_num,1),
                             'vectors': temp_vh.T[:,:eig_num],
                             'begin': data.ix[i].name,
                             'end': data.ix[window_length+i].name})

    return eig_list




def procrustes(target, source):
    """
    Performs a procrustes rotation of source into target

    Args:
    target -- n x k numpy array of vectors
    source -- n x k numpy array of vectors

    Returns:
    R -- k x k numpy array that is a rotation matrix
    """

    if source.shape != target.shape:
        raise RuntimeError('source, target arrays have different dimensions')

    A = source
    B = target
    # TODO(alex): See if we need to flip signs manually for accuracy
    M = numpy.dot(A.T, B)
    u,s,vh = numpy.linalg.svd(M)

    R = numpy.dot(u, vh)

    return R




def prin_angles(p, q, num_angles=None):
    """
    Calculates the principal angles between the vector subspaces p and q

    Args:
    p -- n x k numpy array of n-dimensional vectors
    q -- n x k numpy array of n-dimensional vectors
    num_angles -- An integer denoting the number of principal angles to
                  calculate. Defaults to k if p and q are n x k

    Returns:
    angles -- A k x 1 numpy array of the principal angles
    """

    if p.shape != q.shape:
        raise RuntimeError('p and q have different dimensions')

    if num_angles is None:
        num_angles = p.shape[1]

    angles = numpy.zeros(num_angles)
    for j in xrange(num_angles):
        angles[j] = numpy.arccos(numpy.dot(p[:,j], q[:,j]) / (numpy.dot(p[:,j],p[:,j]) * numpy.dot(q[:,j],q[:,j])))

    return angles.reshape(num_angles,1)




def print_vecs(vecs):
    """
    Prints out the vectors in vecs in a pretty format

    Args:
    vecs -- A numpy array of vectors

    Returns:
    Nothing
    """

    n,k = vecs.shape
    for i in range(n):
        print(' '.join(map(lambda x: '{:7.4f}'.format(x), vecs[i,:])))
    return None




def frob_norm(A, B):
    """
    Returns the Frobenius norm of A - B

    Args:
    A -- A numpy array
    B -- A numpy array

    Returns:
    frob -- The Frobenius norm of the difference between A and B
    """

    frob = numpy.linalg.norm(A - B)

    return frob




def angle_measure(angles):
    """
    Calculates the angular distance

    Args:
    angles -- k x 1 numpy vector of angles

    Returns:
    dist -- A float that represents the total angular distance
    """

    dist = numpy.sqrt(reduce(lambda x,y: x + y, map(lambda x: x*x, angles)))

    return dist
