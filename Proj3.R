graphics.off()
# ====================================================================
# Group 7 - Practical 3
# Team Members:
# Jiayi Qu (s2936988)
# Ting Wu (s2988142)
# Zimo Xiao (s2972405)

# GitHub repository:
# https://github.com/extended-statistical-programming/Practical3.git

# Overview:
# This practical estimates changepoints in Poisson count data using
# a smoothed LASSO penalty. We implement the objective, gradient and
# Hessian, and use BFGS optimization to estimate the mean function with
# an initial value of lambda = 1.
# Then we choose the best penalty value by minimizing mBIC and use the
# corresponding fitted model to estimate the mean function and
# effective changepoint positions.
# Parametric bootstrapping is then used to estimate uncertainty associated
# with the mean function from the best fitted model,
# correct bias and construct 95% confidence intervals.

# Contribution Statement:
# All team members contributed roughly equally (~33.3% each).
# 1. Jiayi Qu: Implemented the penalized objective, gradient and
#    Hessian, tested the derivatives and performed the initial fit with 
#    lambda = 1. Wrote the overview.
# 2. Ting Wu: 
# 3. Zimo Xiao: 

# Collaborative Efforts:
# All parts of the practical were thoroughly discussed as a group.
# Each team member's code was reviewed by the other two after being
# committed to GitHub.

# ========================FUNCTIONS ================================
poisson_lasso_objective <- function(beta,
                                    y,
                                    lambda = 1,
                                    eps = 1e-6) {
  # Evaluate the smoothed LASSO-penalized Poisson objective.
  # Inputs:
  # beta - The parameter vector of length n.
  # y - Observed data vector.
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
  # Inputs:
  # beta - The parameter vector of length n.
  # y - Observed data vector.
  # lambda - The penalty parameter λ (default: 1).
  # eps - The smoothing parameter ϵ for approximating the absolute function
  # (default: 1e-6).
  # Outputs:
  # Gradient vector with respect to beta.
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
  # y_test - The test data vector.
  # h - Finite-difference step size (default: 1e-6).
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
  # Inputs:
  # beta - The parameter vector of length n.
  # y - Observed data vector.
  # lambda - The penalty parameter λ (default: 1).
  # eps - The smoothing parameter ϵ for approximating 
  # the absolute function (default: 1e-6).
  # Outputs:
  # Hessian matrix with respect to beta.
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
  # Inputs:
  # beta_test - The test parameter vector of length n.
  # y_test - The test data vector.
  # h - Finite-difference step size (default: 1e-6).
  # Outputs:
  # The maximum absolute difference between the
  # numerical and analytical Hessian matrices.
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

fit_poisson_lasso <- function(y, lambda, beta_start) {
  # Fit the model for one penalty value and score how good the fit is.
  # Inputs:
  # y - The observed data vector.
  # lambda - The penalty parameter lambda (larger = fewer jumps allowed).
  # beta_start - Starting values for beta (length n).
  # Outputs:
  # A list with:
  # beta - The fitted parameters.
  # mu - The fitted mean at each time point.
  # convergence - 0 if the optimizer finished successfully.
  # m_tilde - The effective number of changepoints.
  # mbic - The mBIC score (lower is better).
  
  fit <- optim(par = beta_start,
               fn = poisson_lasso_objective,
               gr = poisson_lasso_grad,
               lambda = lambda,
               y = y,
               method = "BFGS",
               control = list(maxit = 5000, reltol = 1e-9))
  
  # Warn if the optimizer did not finish, as the fit may be unreliable.
  if (fit$convergence != 0)
    warning("BFGS did not converge for lambda = ", signif(lambda, 4))
  
  beta <- fit$par
  mu <- exp(cumsum(beta))
  
  # EDF is a soft count of parameters: a clear jump counts about 1,
  # a jump shrunk to zero counts about 0.
  # beta_1 (the starting level) also counts 1, so subtract 1 to get the
  # effective number of changepoints.
  H_lambda <- poisson_lasso_hess(beta, y, lambda = lambda)
  H_0 <- poisson_lasso_hess(beta, y, lambda = 0)
  m_tilde <- sum(diag(solve(H_lambda, H_0))) - 1
  
  # Calculate mBIC using the log-likelihood without the LASSO penalty.
  # This balances goodness of fit against having too many changepoints.
  loglik <- sum(dpois(y, mu, log = TRUE))
  mbic <- 3 * m_tilde * log(length(y)) - 2 * loglik
  
  return(list(beta = beta,
              mu = mu,
              convergence = fit$convergence,
              m_tilde = m_tilde,
              mbic = mbic))
}


# ==================== MAIN ANALYSIS ===================================
# Simulate Poisson changepoint data.
set.seed(12345)
n <- 100
time <- 1:n
# cpt positions (with tau_0 and tau_{m+1})
tau_true <- c(1, 30, 70, n + 1) 
# number of cpts (m)
m_true <- length(tau_true) - 2
# segment parameters (theta_1, theta_2, theta_3)
theta_true <- c(10, 2, 7) 
## mean function
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

# Check the gradient and Hessian using finite differences.
grad_error <- first_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))
hess_error <- second_derivative_test(c(log(2), 0, log(5), 0), c(1, 2, 3, 4))
print(c(gradient = grad_error, hessian = hess_error))
# The small differences support the analytical derivatives.

# Fit the initial model using BFGS with lambda = 1.
beta_initial <- numeric(n)
beta_initial[1] <- log(mean(y))
fit <- optim(par = beta_initial, 
             fn = poisson_lasso_objective, 
             gr = poisson_lasso_grad,
             lambda = 1,
             y = y,
             method = "BFGS",
             control = list(maxit = 5000, reltol = 1e-9))
# Check the algorithm convergence and the maximum absolute gradient.
if (fit$convergence != 0)
  warning("Initial BFGS fit did not converge")
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
      col = "red", 
      lwd = 2, 
      type = "s")
abline(v = tau_true[-c(1, length(tau_true))], lty = "dashed")
legend("bottomleft",
       legend = c("True mean function", "Estimated mean function"),
       col = c("black", "red"),
       lwd = 2,
       bty = "n",
       cex = 0.6)

# Try a grid of lambda values and choose the one with the lowest mBIC.
log_lambda <- seq(-1, 5, by = 0.05)
len_log_lambda <- length(log_lambda)
mbic <- numeric(len_log_lambda)
m_tilde <- numeric(len_log_lambda)
convergence <- numeric(len_log_lambda)

beta_start <- beta_initial
best_fit <- NULL

# Record the time taken for the grid search.
grid_time <- system.time(
  for (i in seq_along(log_lambda)) {
    grid_fit <- fit_poisson_lasso(y,
                                  exp(log_lambda[i]),
                                  beta_start)
    
    mbic[i] <- grid_fit$mbic
    m_tilde[i] <- grid_fit$m_tilde
    convergence[i] <- grid_fit$convergence
    
    # Remember the best fit so far, so part 5 can reuse it.
    if (mbic[i] == min(mbic[1:i]))
      best_fit <- grid_fit
    
    # Start the next fit from the current solution to save time.
    beta_start <- grid_fit$beta
  }
)

# Check the elapsed time.
grid_time["elapsed"]

# Check whether every fit converged successfully.
all(convergence == 0)

# Select the lambda value with the smallest mBIC.
lambda_hat <- exp(log_lambda[which.min(mbic)])
lambda_hat

# All fits converged in about 1-2 seconds. The selected lambda is about 21.12
# (log(lambda) = 3.05), which lies inside the search grid rather than
# at one of its boundaries.


# Plot mBIC and the effective number of changepoints against log(lambda).
par(mfrow = c(1, 2))

plot(log_lambda,
     mbic,
     type = "l",
     lwd = 2,
     xlab = "log(lambda)",
     ylab = "mBIC",
     main = "mBIC")

abline(v = log(lambda_hat),
       lty = "dashed",
       col = "red")

plot(log_lambda,
     m_tilde,
     type = "l",
     lwd = 2,
     xlab = "log(lambda)",
     ylab = "Effective no. of changepoints",
     main = "Effective changepoints")

abline(v = log(lambda_hat),
       lty = "dashed",
       col = "red")

par(mfrow = c(1, 1))

# Small lambda values allow many changepoints, while large lambda values
# penalise changes more strongly and therefore give fewer changepoints.
# The red dashed line marks the selected lambda.


# 5. Use the best fit to estimate the changepoints.

# The fit at the best lambda was saved above, so no refitting is needed.
best_fit$convergence

beta_hat <- best_fit$beta
mu_hat <- best_fit$mu

# Estimate the number of changepoints by rounding m_tilde down.
m_hat <- floor(best_fit$m_tilde)

# The estimated changepoints are the times with the largest |beta_k|.
# beta_1 is the starting level rather than a changepoint, so it is excluded.
# After removing beta_1, add 1 to convert positions back to time points.
# seq_len(m_hat) is empty when m_hat = 0, avoiding incorrect indexing.
tau_hat <- sort(
  order(abs(beta_hat[-1]), decreasing = TRUE)[seq_len(m_hat)] + 1
)

if (m_hat == 0)
  message("No changepoints detected at lambda_hat.")

# Display the effective and estimated numbers of changepoints.
c(m_tilde = best_fit$m_tilde,
  m_hat = m_hat)

# Display the estimated changepoint locations.
tau_hat
# 4 changepoints are found, at times 30, 69, 70 and 72 (truth: 30 and 70).
# The jump at 30 is found exactly; the jump at 70 is split into three
# small steps.


# Plot the data, true and estimated mean functions, and changepoints.
plot(time,
     y,
     pch = 16,
     col = "grey",
     ylim = c(0, 1.3 * max(y)),
     xlab = "Time",
     ylab = "Count",
     main = "Best fitted model")

# True mean function.
lines(mu_true,
      col = "black",
      lwd = 2,
      type = "s")

# Estimated mean function.
lines(mu_hat,
      col = "red",
      lwd = 2,
      type = "s")

# True changepoints.
abline(v = tau_true[-c(1, length(tau_true))],
       lty = "dashed")

# Estimated changepoints.
abline(v = tau_hat,
       lty = "dotted",
       col = "red")

legend("top",
       legend = c("Data",
                  "True mean",
                  "Estimated mean",
                  "True changepoints",
                  "Estimated changepoints"),
       col = c("grey",
               "black",
               "red",
               "black",
               "red"),
       pch = c(16, NA, NA, NA, NA),
       lty = c(NA, 1, 1, 2, 3),
       lwd = c(NA, 2, 2, 1, 1),
       ncol = 3,
       bty = "n",
       cex = 0.8)

# Parametric bootstrap and uncertainty estimation.
set.seed(55555)
b <- 200
# Initialize storage for bootstrap estimates.
sample_matrix <- matrix(data = NA, nrow = b, ncol = n)

i <- 1
j <- 0
while (i <= b) {
  # Repeat until b converged bootstrap fits are obtained.
  # Simulate Poisson data using the fitted mean.
  y_sim <- rpois(n, mu_hat)
  fit_boot <- optim(
    par = beta_hat,
    fn = poisson_lasso_objective,
    gr = poisson_lasso_grad,
    y = y_sim,
    lambda = lambda_hat,
    method = "BFGS",
    control = list(maxit = 5000, reltol = 1e-9)
  )
  # Store the fitted mean only if BFGS reports convergence. 
  if (fit_boot$convergence == 0) {
    eta <- cumsum(fit_boot$par)
    mu <- exp(eta)
    sample_matrix[i, ] <- mu
    i <- i + 1
  }
  # Count all fitting attempts.
  j <- j + 1 
}

# Count the bootstrap fits that did not converge.
# All 200 fits converged in the test using seed 55555.
# No attempt limit is imposed, so repeated failures could
# prevent the loop from terminating.
# A maximum number of attempts could be added if needed.
fail_fits <- j - b
print(fail_fits)

# Average the bootstrap estimates at each time point.
colmean <- colMeans(sample_matrix)

# Compare the true, fitted and bootstrap-average mean by plotting a graph.
par(mar = c(5, 4, 4, 13))
plot(time, y,
     main = "Bootstrap Mean Comparison",
     ylim = range(c(y, mu_true, mu_hat, colmean)),
     pch = 16,
     col = "grey",
     xlab = "Time",
     ylab = "Count")
lines(time, mu_true,
      type = "s",
      col = "black",
      lwd = 2)
lines(time, mu_hat,
      type = "s",
      col = "red",
      lwd = 2)
lines(time, colmean,
      type = "s",
      col = "blue",
      lwd = 2)
legend("topright",
       inset = c(-0.31, 0),
       legend = c("Data", "True Mean", 
                  "Fitted Mean", "Bootstrap Mean"),
       col = c("grey", "black", "red", "blue"),
       pch = c(16, NA, NA, NA),
       lty = c(NA, 1, 1, 1),
       xpd = NA,
       bty = "n",
       cex = 0.8)
print(mean(abs(mu_hat - mu_true)))
print(mean(abs(colmean - mu_true)))
# The fitted mean captures the main changes near times 30 and 70. 
# The bootstrap average is smoother near the changepoints
# and deviates further from the true mean in parts of the series.
# This reflects bias from penalisation and 
# motivates the bootstrap bias correction.

# Correct the fitted mean using the bootstrap bias estimate.
correctmu <- 2 *mu_hat - colmean

# Compute the pointwise 95% basic bootstrap confidence intervals.
CI <- apply(sample_matrix, 2, FUN = quantile, probs = c(0.025,0.975))
QL <- CI[1,]
QU <- CI[2,]
CQL <- 2* mu_hat - QU
CQU <- 2* mu_hat - QL

# Check that the lower bounds do not exceed the upper bounds.
print(all(CQL <= CQU))

# Compare the true, fitted and bias-corrected mean functions.
# Add the pointwise 95% confidence bounds.
par(mar = c(5, 4, 4,15))
plot(time, y,
     main = "Bias-corrected Mean and Confidence Intervals",
     ylim = range(c(y, mu_true, mu_hat, 
                    correctmu, CQL, CQU)),
     pch = 16,
     col = "grey",
     xlab = "Time",
     ylab = "Count")
lines(time, mu_true,
      type = "s",
      col = "black",
      lwd = 2)
lines(time, mu_hat,
      type = "s",
      col = "red",
      lwd = 2)
lines(time, correctmu,
      type = "s",
      col = "green",
      lwd = 2)
lines(time, CQL,
      type = "s",
      col = "blue",
      lty = 2)
lines(time, CQU,
      type = "s",
      col = "blue",
      lty = 2)
legend("topright",
       inset = c(-0.45, 0),
       legend = c("Data", "True Mean", "Fitted Mean", 
                  "Bias-Corrected Mean", 
                  "95% Bias-corrected CI"),
       col = c("grey", "black", "red", "green", "blue"),
       pch = c(16, NA, NA, NA, NA),
       lty = c(NA, 1, 1, 1, 2),
       xpd = NA,
       bty = "n",
       cex = 0.8)
# Bias correction may overshoot near changepoints when
# the bootstrap average differs substantially from the fitted mean.


print(sum(CQL < 0))
# The basic bootstrap CI has one negative lower bound,
# although the Poisson mean must be positive.
# This illustrates a limitation of the interval method.

print(mean(abs(correctmu - mu_true)))
# Bias correction reduces the MAE for this simulated dataset.