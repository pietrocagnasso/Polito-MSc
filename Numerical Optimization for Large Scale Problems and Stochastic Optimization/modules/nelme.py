from .armijo import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
# https://www.researchgate.net/publication/225691623_Implementing_the_Nelder-Mead_simplex_algorithm_with_adaptive_parameters

def nelmead(f, x_0, adaptive=False, max_iter=10000, tol=1e-5, verbose=True):
    N = x_0.shape[0]
    
    if N ==2:
        S_seq = []
        fS_seq = []
    
    if adaptive:
        rho = 1
        chi = 1 + 2 / N
        gamma = 0.75 - 1 / (2 * N)
        sigma = 1 - 1 / N
    else:
        rho, chi, gamma, sigma = 1, 2, 0.5, 0.5
    
    # simplex initialization
    nonzero_delta = 0.005
    zero_delta = 0.00025    
    
    S = np.empty((N+1, N))
    S[0] = x_0.T
    
    for i in range(N):
        x = np.array(x_0, copy=True)
        
        if x[i] == 0:
            x[i] = zero_delta
        else:
            x[i] = (1 + nonzero_delta) * x[i]
            
        S[i+1] = x.T
    
    fS = np.empty((N+1,))
    for i in range(S.shape[0]):
        fS[i] = f(S[i])
       
    if N == 2:
        S_seq.append(S)
        fS_seq.append(fS)
    
    k = 0
    stop = False
    try:
        while k < max_iter and not stop:
            k += 1
            if k % 10000 == 0:
                print(f"Iteration: {k}")

            # sort the simplex vertex based on function evaluation
            ind = np.argsort(fS)
            S = np.take(S, ind, 0)
            fS = np.take(fS, ind, 0)

            # compute the center of top n vertex
            x_bar = np.add.reduce(S[:-1], 0) / N

            # reflection
            x_r = x_bar + rho * (x_bar - S[-1])
            f_r = f(x_r)

            if f_r < fS[0]:
                # expansion
                x_e = x_bar + chi * (x_r - x_bar)
                f_e = f(x_e)

                if f_e < f_r:
                    S[-1] = x_e
                    fS[-1] = f_e
                else:
                    S[-1] = x_r
                    fS[-1] = f_r
            else:
                if f_r < fS[-2]:
                    S[-1] = x_r
                    fS[-1] = f_r
                else:
                    shrink = False
                    if f_r < fS[-1]:
                        # outside contraction
                        x_oc = x_bar + gamma * (x_r - x_bar)
                        f_oc = f(x_oc)

                        if f_oc <= f_r:
                            S[-1] = x_oc
                            fS[-1] = f_oc
                        else:
                            shrink = True
                    else:
                        # inside contraction
                        x_ic = x_bar - gamma * (x_r - x_bar)
                        f_ic = f(x_ic)

                        if f_ic < fS[-1]:
                            S[-1] = x_ic
                            fS[-1] = f_ic
                        else:
                            shrink = True

                    if shrink:
                        # shrinkage
                        for i in range(1, x_0.shape[0]+1):
                            S[i] = S[0] + sigma * (S[i] - S[0])
                            fS[i] = f(S[i])


            # stopping criterion
            stop = (np.max(np.ravel(np.abs(S[1:] - S[0]))) < tol and
                       np.max(np.abs(fS[0] - fS[1:])) < tol)
            
            if N == 2:
                S_seq.append(S)
                fS_seq.append(fS)

        ind = np.argsort(fS)
        S = np.take(S, ind, 0)
        fS = np.take(fS, ind, 0)

        if verbose:
            if x.shape == (2, 1):
                print(f"""Starting point: {x_0.T}
Solution point: {S[0]}
Function value in the solution: {fS[0]}
Iterations needed to reach the solution: {k}""")
            else:
                print(f"""Function value in the solution: {fS[0]}
Iterations needed to reach the solution: {k}""")
    except KeyboardInterrupt:
        print(f"Run interrupted manually at iteration: {k}")
    except Exception as e:
        print("Error: ", e)
    finally:
        if N == 2:
            return S_seq, fS_seq, k, stop
        else:
            return S, fS, k, stop