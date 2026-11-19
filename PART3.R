source("Part1.R")
warnings()
source("Part2.R")

# 6
# Set a random seed for reproducible bootstrap simulations.
set.seed(55555)
b <- 200

# Initialize storage for bootstrap estimates.
sample_matrix <- matrix(data = NA, nrow = b, ncol = 100)


i <- 1
j <- 1
while (i <= b) {
  # Repeat until b converged bootstrap fits are obtained.
  # Simulate Poisson data using the fitted mean.
  y_sim <- rpois(100, mu_hat)
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
  if (fit_boot$convergence == 0){
    eta <- cumsum(fit_boot$par)
    mu <- exp(eta)
    sample_matrix[i,] <- mu
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
fail_fits <- (j - 1) - b
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



#  7
# Check that the lower bounds do not exceed the upper bounds.
print(all(CQL <= CQU))

# Compare the true, fitted and bias-corrected mean functions.
# Add the pointwise 95% confidence bounds.
par(mar = c(5, 4, 4, 15))
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
      col= "green",
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