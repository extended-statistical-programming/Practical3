## 1 
# Simulate Poission changepoint dataset
set.seed(12345)
n <- 100
time <- 1:n
tau_true <- c(1, 30, 70, n + 1)
m_true <- length(tau_true) - 2
theta_true <- c(10, 2, 7)
mu_true <- numeric(n)
for (i in seq_along(theta_true))
  mu_true[tau_true[i]:(tau_true[i + 1] - 1)] <- theta_true[i]
y <- rpois(n, lambda = mu_true)
plot(time, y, pch = 16, col = "grey")
lines(mu_true,
      col = "black",
      lwd = 2,
      type = "s")
abline(v = tau_true[-c(1, length(tau_true))], lty = "dashed")

## 2
poission_lasso_objective <- function(beta,
                                     y,
                                     lambda = 1,
                                     eps = 1e-6) {
  # Evaluate the objective in Eq.(6)
  # Inputs:
  # beta - The parameter vector of length n.
  # y - The data vector
  # lambda - The penalty parameter λ.
  # eps - The smoothing parameter ϵfor approximating the absolute function.
  # Outputs:
  # The objective value
  eta <- cumsum(beta)
  # Calculate the negative log-likelihood.
  # lgamma(y + 1) computes more stably than log(factorial(y)).
  nll <- sum(exp(eta) - y * eta + lgamma(y + 1))
  lasso_penalty <- lambda * sum(sqrt(beta[-1]^2 + eps))
  return(nll + lasso_penalty)
}


poission_lasso_grad <- function(beta,
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
  # beta - The test parameter vector of length n.
  # y - The test data vector
  # Outputs:
  # Maximum absolute difference between corresponding results calculated 
  # by two methods
  grad_fd <- numeric(length(beta_test))
  for (j in seq_along(beta_test)) {
    beta_plus <- beta_test
    beta_minus <- beta_test
    beta_plus[j] <- beta_plus[j] + h
    beta_minus[j] <- beta_minus[j] - h
    C_plus <- poission_lasso_objective(beta_plus, y_test)
    C_minus <- poission_lasso_objective(beta_minus, y_test)
    grad_fd[j] <- (C_plus - C_minus) / (2 * h)
  }
  grad_test <- poission_lasso_grad(beta_test, y_test)
  return(max(abs(grad_fd - grad_test)))
}
# test
first_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))


poission_lasso_hess <- function(beta,
                                y,
                                lambda = 1,
                                eps = 1e-6) {
  # Evaluate the second derivative (Hessian matrix) of objective.
  eta <- cumsum(beta)
  mu <- exp(eta)
  # Calculate Hessian of the negative log-likeliood.
  n <- length(beta)
  index <- outer(1:n, 1:n, pmax)
  s <- rev(cumsum(rev(mu)))
  H_nll <- matrix(s[index], n, n)
  # Calculate Hessian of the smoothed LASSO penalty
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
    grad_plus <- poission_lasso_grad(beta_plus, y_test)
    grad_minus <- poission_lasso_grad(beta_minus, y_test)
    H_fd[,j] <- (grad_plus - grad_minus) / (2 * h)
  }
  H_test <- poission_lasso_hess(beta_test, y_test)
  return(max(abs(H_fd - H_test)))
}
# test
second_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))


# 3. 