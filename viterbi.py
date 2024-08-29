import numpy as np
from numba import jit

@jit(nopython=True)
def viterbi_log(O, A, Pi, B):
    """Viterbi algorithm (log variant) for solving the HMM decoding problem

    Args:
        A  (np.ndarray): State transition probability matrix of dimension I x I
        Pi (np.ndarray): Initial state distribution  of dimension I
        B  (np.ndarray): Output probability matrix of dimension I x K
        O  (np.ndarray): Observation sequence of length N

    Returns:
        Q_opt (np.ndarray): Optimal state sequence of length N
        Tr_log (np.ndarray): Accumulated log probability matrix
        Bt (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    Pi_log = np.log(Pi + tiny)
    B_log = np.log(B + tiny)

    # Initialize trellis and backtracking matrices
    Tr_log = np.zeros((I, N))
    Bt = np.zeros((I, N-1)).astype(np.int32)
    Tr_log[:, 0] = Pi_log + B_log[:, O[0]]

    # Compute trellis and backtracking in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + Tr_log[:, n-1]
            Tr_log[i, n] = np.max(temp_sum) + B_log[i, O[n]]
            Bt[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    Q_opt = np.zeros(N).astype(np.int32)
    Q_opt[-1] = np.argmax(Tr_log[:, -1])
    for n in range(N-2, -1, -1):
        Q_opt[n] = Bt[int(Q_opt[n+1]), n]

    return Q_opt, Tr_log, Bt
