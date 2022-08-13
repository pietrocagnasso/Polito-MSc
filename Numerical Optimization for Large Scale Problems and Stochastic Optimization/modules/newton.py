from .armijo import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

def nm(f, gradf, hessf, x_0, fd=False, alpha_0=1, max_iter=10000, tol=1e-5):
    x_seq = []
    f_seq = []
    
    x_k = x_0
    if fd:
        gradf_k = gradf(x_k, f)
    else:
        gradf_k = gradf(x_k)
    
    x_seq.append(x_k)
    f_seq.append(f(x_k))
    
    try:
        k = 0
        stop = False
        while k < max_iter and not stop:
            k += 1
            if k % 25000 == 0:
                print(f"Iteration: {k}")

            # compute p
            if fd:
                p_k = solve(-hessf(x_k, f), gradf_k)
            else:
                p_k = solve(-hessf(x_k), gradf_k)

            # line search for alpha
            alpha_k = alpha_0
            t = 0
            while not is_armijo_satisfied(f, f(x_k), gradf_k, x_k, p_k, alpha_k) and t < 100:
                t += 1
                alpha_k = armijo_update(alpha_k)

            # update x
            newx_k = x_k + alpha_k * p_k

            # update gradf
            if fd:
                newgradf_k = gradf(newx_k, f)
            else:
                newgradf_k = gradf(newx_k)

            # stopping criterion
            stop = norm(newgradf_k) < tol

            x_k = newx_k; gradf_k = newgradf_k
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

    

def fd_nm(f, gradf, hessf, x_0, fd=True, alpha_0=1, max_iter=10000, tol=1e-5):
    return nm(f, gradf, hessf, x_0, fd=fd, alpha_0=alpha_0, max_iter=max_iter, tol=tol)



def force_term(grad_norm, tp):
    if tp == "suplin":
        if np.sqrt(grad_norm) < 0.5:
            return np.sqrt(grad_norm)
        else:
            return 0.5
    elif tp == "quad":
        if grad_norm < 0.5:
            return grad_norm
        else:
            return 0.5
        
        

def inm(f, gradf, hessf, x_0, tp="suplin", alpha_0=1, max_iter=10000, tol=1e-5):
    x_seq = []
    f_seq = []
    
    x_k = x_0
    gradf_k = gradf(x_k)
    
    x_seq.append(x_k)
    f_seq.append(f(x_k))
    
    k = 0
    stop = False
    try:
        while k < max_iter and not stop:
            k += 1
            if k % 10000 == 0:
                print(f"Iteration: {k}")

            # compute p
            eta_k = force_term(norm(gradf_k), tp)
            p_k = cg(hessf(x_k), -gradf_k, tol=min(2e-2, eta_k*norm(gradf_k)))[0].reshape(-1, 1)

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

            # stopping criterion
            stop = norm(newgradf_k) < tol

            x_k = newx_k; gradf_k = newgradf_k
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