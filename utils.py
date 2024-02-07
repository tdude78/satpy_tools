import numpy as np

def get_cov_samples(covar:np.ndarray, mean=None, N=1):
    if mean is None:
        mean = np.zeros(covar.shape[0])
    N       = int(N)
    try:
        s_mat   = np.linalg.cholesky(covar)
    except np.linalg.LinAlgError:
        s_mat   = np.linalg.cholesky(covar + 1e-6*np.eye(covar.shape[0]))
    size    = covar.shape[0]
    samples = np.zeros((N,size))
    for i, _ in enumerate(range(N)):
        samples[i,:] = np.dot(s_mat,np.random.normal(size=size)) + mean
    return samples

def get_orthogonal_vectors(v):
    # Normalize the given vector
    v_norm = v / np.linalg.norm(v)

    # Create the second vector
    v2 = np.cross(v_norm, [0, 0, 1])
    if np.linalg.norm(v2) == 0:
        v2 = np.cross(v_norm, [0, 1, 0])
    if np.linalg.norm(v2) == 0:
        v2 = np.cross(v_norm, [1, 0, 0])
    v2_norm = v2 / np.linalg.norm(v2)

    # Create the third vector
    v3 = np.cross(v_norm, v2_norm)
    v3_norm = v3 / np.linalg.norm(v3)

    return v_norm, v2_norm, v3_norm