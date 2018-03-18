plotCutoffs <- function(data, cutoffs) {
  
  plot(data,
       main = "Tröskelvärde för binarisering av annoteringar",
       xlab = "Genomsnittligt läsbarhetsbetyg", ylab = "Källkodsdokumentets täthetsvärde",
       col = "#4C72B0",
       bg = "#EAEAF2")
  
  lw <- 2
  abline(v = cutoffs, col = c("#CB6D72", "#73B583"), lty = 2:1, lw = lw)
  
  grid(col = "gray")
  
  text(cutoffs[1], 0.6, round(cutoffs[1], digits = 2), pos = 2, offset = 1, lw = lw)
  text(cutoffs[2], 0.6, round(cutoffs[2], digits = 2), pos = 4, offset = 1, lw = lw)
  
  legend(1.5, 0.6, 
         c("50% sannolikhet att datapunkt\ntillhör den mindre populationen\n",
           "42% sannolikhet att datapunkt\ntillhör den mindre populationen"),
         col = c("#CB6D72", "#73B583"),
         lty = 2:1)
}