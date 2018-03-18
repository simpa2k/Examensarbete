# Adapted from an original solution taken from:
# https://stats.stackexchange.com/questions/57993/how-to-explain-how-i-divided-a-bimodal-distribution-based-on-kernel-density-esti/72005#72005

library(mixtools)

set.seed(2969)

applyNormalMixEM <- function(x) {
  
  model <- normalmixEM(x=x, k=2)
  index.lower <- which.min(model$mu)  # Index of component with lower mean
  
  find.cutoff <- function(proba=0.5, i=index.lower) {
    ## Cutoff such that Pr[drawn from bad component] == proba
    f <- function(x) {
      proba - (model$lambda[i]*dnorm(x, model$mu[i], model$sigma[i]) /
                 (model$lambda[1]*dnorm(x, model$mu[1], model$sigma[1]) + model$lambda[2]*dnorm(x, model$mu[2], model$sigma[2])))
    }
    return(uniroot(f=f, lower=0, upper=5)$root)  # Careful with division by zero if changing lower and upper
  }
  
  cutoffs <- c(find.cutoff(proba = 0.5), find.cutoff(proba = 0.42))
  
  return(cutoffs)
}
