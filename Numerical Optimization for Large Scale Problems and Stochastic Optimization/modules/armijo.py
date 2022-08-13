def is_armijo_satisfied(f, f_k, gradf_k, x_k, p_k, alpha_k, c=1e-4):
    return f(x_k + alpha_k * p_k) <= f_k + (c * alpha_k * gradf_k.T @ p_k)

def armijo_update(alpha_k, rho=0.5):
    return alpha_k * rho
