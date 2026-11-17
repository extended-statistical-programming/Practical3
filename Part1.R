# ===================FUNCTIONS ================================
## 2
poisson_lasso_objective <- function(beta,
                                     y,
                                     lambda = 1,
                                     eps = 1e-6) {
  # Evaluate the objective in Eq.(6)
  # Inputs:
  # beta - The parameter vector of length n.
  # y - The data vector
  # lambda - The penalty parameter λ (default: 1).
  # eps - The smoothing parameter ϵ for approximating the absolute function
  # (default: 1e-6).
  # Outputs:
  # The objective value.
  eta <- cumsum(beta)
  # Calculate the negative log-likelihood.
  # lgamma(y + 1) computes more stably than log(factorial(y)).
  nll <- sum(exp(eta) - y * eta + lgamma(y + 1))
  lasso_penalty <- lambda * sum(sqrt(beta[-1]^2 + eps))
  return(nll + lasso_penalty)
}

poisson_lasso_grad <- function(beta,
                                y,
                                lambda = 1,
                                eps = 1e-6) {
  # Evaluate the gradient of the objective with respect to beta.
  eta <- cumsum(beta)
  mu <- exp(eta)
  grad_nll <- rev(cumsum(rev(mu - y)))
  grad_penalty <- c(0, lambda * beta[-1] / sqrt(beta[-1]^2 + eps))
  return(grad_nll + grad_penalty)
}

first_derivative_test <- function(beta_test, y_test, h = 1e-6) {
  # Test the gradient using finite differences.
  # Inputs:
  # beta_test - The test parameter vector of length n.
  # y_test - The test data vector
  # Outputs:
  # The maximum absolute difference between the
  # numerical and analytical gradients.
  grad_fd <- numeric(length(beta_test))
  for (j in seq_along(beta_test)) {
    beta_plus <- beta_test
    beta_minus <- beta_test
    beta_plus[j] <- beta_plus[j] + h
    beta_minus[j] <- beta_minus[j] - h
    C_plus <- poisson_lasso_objective(beta_plus, y_test)
    C_minus <- poisson_lasso_objective(beta_minus, y_test)
    grad_fd[j] <- (C_plus - C_minus) / (2 * h)
  }
  grad_test <- poisson_lasso_grad(beta_test, y_test)
  return(max(abs(grad_fd - grad_test)))
}

poisson_lasso_hess <- function(beta,
                                y,
                                lambda = 1,
                                eps = 1e-6) {
  # Evaluate the Hessian matrix of the objective.
  eta <- cumsum(beta)
  mu <- exp(eta)
  # Calculate Hessian of the negative log-likelihood.
  n <- length(beta)
  index <- outer(1:n, 1:n, pmax)
  s <- rev(cumsum(rev(mu)))
  H_nll <- matrix(s[index], n, n)
  # Calculate Hessian of the smoothed LASSO penalty.
  penalty_diag <- c(0, lambda * eps / (beta[-1]^2 + eps)^(3 / 2))
  return(H_nll + diag(penalty_diag))
}

second_derivative_test <- function(beta_test, y_test, h = 1e-6) {
  # Test the Hessian using finite differences.
  n <- length(beta_test)
  H_fd <- matrix(0, n, n)
  for (j in seq_along(beta_test)) {
    beta_plus <- beta_test
    beta_minus <- beta_test
    beta_plus[j] <- beta_plus[j] + h
    beta_minus[j] <- beta_minus[j] - h
    grad_plus <- poisson_lasso_grad(beta_plus, y_test)
    grad_minus <- poisson_lasso_grad(beta_minus, y_test)
    H_fd[,j] <- (grad_plus - grad_minus) / (2 * h)
  }
  H_test <- poisson_lasso_hess(beta_test, y_test)
  return(max(abs(H_fd - H_test)))
}

# ================== MAIN ANALYSIS =====================================
# 1. Simulate Poisson changepoint data.
set.seed(12345)
n <- 100
time <- 1:n
tau_true <- c(1, 30, 70, n + 1) ## cpt positions (with tau_0 and tau_{m+1})
m_true <- length(tau_true) - 2 ## number of cpts (m)
theta_true <- c(10, 2, 7) ## segment parameters (theta_1, theta_2, theta_3)
mu_true <- numeric(n) ## mean function
for (i in seq_along(theta_true))
  mu_true[tau_true[i]:(tau_true[i + 1] - 1)] <- theta_true[i]
y <- rpois(n, lambda = mu_true)
plot(time, y, pch = 16, col = "grey")
lines(mu_true,
      col = "black",
      lwd = 2,
      type = "s")
abline(v = tau_true[-c(1, length(tau_true))], lty = "dashed")

# 2. Check the gradient and Hessian using finite differences.
first_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))
second_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))
# The small differences support the analytical derivatives.

# 3. A preliminary sanity check.
# Fit the model using BFGS.
beta_initial <- numeric(n)
beta_initial[1] <- log(mean(y))
fit <- optim(par = beta_initial, 
      fn = poisson_lasso_objective, 
      gr = poisson_lasso_grad,
      lambda = 1,
      y = y,
      method = "BFGS",
      control = list(maxit = 1000, reltol = 1e-9))
# Check the algorithm convergence (0 indicates success) and 
# the maximum absolute gradient.
fit$convergence
max(abs(poisson_lasso_grad(fit$par, y)))
# BFGS converged successfully and the gradient is close to zero.

# Evaluate the mean function from the parameter estimates.
mu_hat <- exp(cumsum(fit$par))
# Plot the data (y) and add both the true (mu_true) and 
# estimated (mu_hat) mean functions.
# The diagram shows that the estimated mean function
# captures the main changes near times 30 and 70,
# with some fluctuations within each segment.
plot(time, y, pch = 16, col = "grey")
lines(mu_true,
      col = "black",
      lwd = 2,
      type = "s")
lines(mu_hat, 
      col = "pink", 
      lwd = 2, 
      type = "s")
abline(v = tau_true[-c(1, length(tau_true))], lty = "dashed")
legend("bottomleft",
       legend = c("True mean function", "Estimated mean function"),
       col = c("black", "pink"),
       lwd = 2,
       bty = "n",
       cex = 0.6)
