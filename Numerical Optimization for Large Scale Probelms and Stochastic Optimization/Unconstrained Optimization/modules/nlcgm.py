from .armijo import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

def fr(f, gradf, x_0, alpha_0=1, max_iter=10000, tol=1e-8):
    x_seq = []
    f_seq = []
    
    x_k = x_0
    gradf_k = gradf(x_k)
    p_k = -gradf_k
    
    x_seq.append(x_k)
    f_seq.append(f(x_k))
    
    k = 0
    stop = False
    try:
        while k < max_iter and not stop:        
            k += 1
            if k % 10000 == 0:
                print(f"Iteration: {k}")

            # line search for alpha
            alpha_k = alpha_0
            t = 0
            while not is_armijo_satisfied(f, f(x_k), gradf_k, x_k, p_k, alpha_k) and t < 100:
                t += 1
                alpha_k = armijo_update(alpha_k)

            # update x
            newx_k = x_k + alpha_k * p_k

            # update gradf
            newgradf_k = gradf(newx_k)

            # compute beta
            beta_k = norm(newgradf_k)**2 / norm(gradf_k)**2

            # update p
            newp_k = -newgradf_k + beta_k * p_k

            # stopping criterion
            stop = norm(newgradf_k) < tol

            x_k = newx_k; gradf_k = newgradf_k; p_k = newp_k
            x_seq.append(x_k)
            f_seq.append(f(x_k))

        f_k = f(x_k)
        if x_0.shape == (2, 1):
            print(f"""Starting point: {x_0.T}
Solution point: {x_k.T}
Function value in the solution: {f_k}
Gradient norm in the solution: {norm(gradf_k)}
Iterations needed to reach the solution: {k}""")
        else:
            print(f"""Function value in the solution: {f_k}
Gradient norm in the solution: {norm(gradf_k)}
Iterations needed to reach the solution: {k}""")   
    except KeyboardInterrupt:
        print(f"Run interrupted manually at iteration: {k}")
    except Exception as e:
        print("Error: ", e)
    finally:
        return x_seq, f_seq, k, stop
    
    
def pr(f, gradf, x_0, alpha_0=1, max_iter=10000, tol=1e-5):
    x_seq = []
    f_seq = []
    
    x_k = x_0
    gradf_k = gradf(x_k)
    p_k = -gradf_k
    
    x_seq.append(x_k)
    f_seq.append(f(x_k))
    
    k = 0
    stop = False
    try:
        while k < max_iter and not stop:
            k += 1
            if k % 10000 == 0:
                print(f"Iteration: {k}")

            # line search for alpha
            alpha_k = alpha_0
            t = 0
            while not is_armijo_satisfied(f, f(x_k), gradf_k, x_k, p_k, alpha_k) and t < 100:
                t += 1
                alpha_k = armijo_update(alpha_k)

            # update x
            newx_k = x_k + alpha_k * p_k

            # update gradf
            newgradf_k = gradf(newx_k)

            # compute beta
            beta_k = (newgradf_k.T @ (newgradf_k - gradf_k)) / (gradf_k.T @ gradf_k)

            # update p
            newp_k = -newgradf_k + beta_k * p_k

            # stopping criterion
            stop = norm(newgradf_k) < tol

            x_k = newx_k; gradf_k = newgradf_k; p_k = newp_k
            x_seq.append(x_k)
            f_seq.append(f(x_k))

        f_k = f(x_k)
        if x_0.shape == (2, 1):
            print(f"""Starting point: {x_0.T}
Solution point: {x_k.T}
Function value in the solution: {f_k}
Gradient norm in the solution: {norm(gradf_k)}
Iterations needed to reach the solution: {k}""")
        else:
            print(f"""Function value in the solution: {f_k}
Gradient norm in the solution: {norm(gradf_k)}
Iterations needed to reach the solution: {k}""")
    except KeyboardInterrupt:
        print(f"Run interrupted manually at iteration: {k}")
    except Exception as e:
        print("Error: ", e)
    finally:
        return x_seq, f_seq, k, stop