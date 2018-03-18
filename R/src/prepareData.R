averageData <- function(x) {
  pruned <- x[, -c(1:2)]
  averaged <- colMeans(pruned)
  
  return(averaged)
}

prepareData <- function(x) {
  averaged <- averageData(x)
  d = density(averaged, bw = 0.08, give.Rkern = FALSE)
  cutoffs <- applyNormalMixEM(d$x)
  
  binarized <- ifelse(averaged < cutoffs[2], 0, 1)
  
  return(list(density = d, binarizedData = binarized, cutoffs = cutoffs))
}