# ================== FUNCTIONS ==========================================

## 4-5
fit_poisson_lasso <- function(y, lambda, beta_start) {
  # Fit the model for one penalty value and score how good the fit is.
  # Inputs:
  # y - The data vector.
  # lambda - The penalty parameter lambda (larger = fewer jumps allowed).
  # beta_start - Starting values for beta (length n).
  # Outputs:
  # A list with:
  # beta - The fitted parameters.
  # mu - The fitted mean at each time point.
  # convergence - 0 if the optimiser finished successfully.
  # m_tilde - The effective number of changepoints.
  # mbic - The mBIC score (lower is better).
  
  fit <- optim(par = beta_start,
               fn = poisson_lasso_objective,
               gr = poisson_lasso_grad,
               lambda = lambda,
               y = y,
               method = "BFGS",
               control = list(maxit = 5000, reltol = 1e-9))
  
  # Warn if the optimiser did not finish, as the fit may be unreliable.
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


# ================== MAIN ANALYSIS ======================================

# 4. Try a grid of lambda values and choose the one with the lowest mBIC.
log_lambda <- seq(-1, 5, by = 0.05)

mbic <- numeric(length(log_lambda))
m_tilde <- numeric(length(log_lambda))
convergence <- numeric(length(log_lambda))

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

# All fits converged in about 1-2 seconds. The selected lambda is about
# 20.09 (log(lambda) = 3), which lies inside the search grid rather than
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

# The fit at the best lambda was saved in part 4, so no refitting is needed.
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

