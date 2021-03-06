library(utils)
library(irr)

data <- annotations[, 2:5]
data <- as.data.frame(sapply(data, as.numeric))
combinations <- (combn(1:ncol(data), 2))

gatherCorrelationDataAsList <- function(kappa, weightedKappa, kendall, pearson, spearman) {
  list(
    c(kappa$value,          kappa$p.value),
    c(weightedKappa$value,  weightedKappa$p.value),
    c(kendall$estimate,     kendall$p.value),
    c(pearson$estimate,     pearson$p.value),
    c(spearman$estimate,    spearman$p.value)
  )
}

performTests <- function(voter1And2, voter1, voter2) {
  gatherCorrelationDataAsList(
    kappa2  (voter1And2),
    kappa2  (voter1And2, weight = 'equal'), # Linear weights
    cor.test(voter1, voter2, method = 'kendall'),
    cor.test(voter1, voter2, method = 'pearson'),
    cor.test(voter1, voter2, method = 'spearman')
  )
}

results <- apply(combinations, 2, function(x) {
  performTests(
    data[, c(x[1], x[2])],
    data[, x[1]],
    data[, x[2]])
})

reduced <- Reduce(function(a, b) {
  lapply(
    seq_along(a),
    function(i) unlist(a[i]) + unlist(b[i])
  )
}, results)

averaged <- Map(function(x) x / ncol(data), reduced)

correlations <- Map(function(x) x[1], averaged)
pValues      <- Map(function(x) x[2], averaged)

df <- do.call(rbind,
              Map(data.frame, c = correlations, p = pValues))

df[, 'c'] <- round(df[, 'c'], 2)
df[, 'p'] <- round(df[, 'p'], 4)

colnames(df) <- c('Korrelation', 'p (Tvåsidigt)')
rownames(df) <- c('Cohens \\(\\kappa\\)', 
                  'Linjärt viktad Cohens \\(\\kappa\\)', 
                  'Kendalls \\(\\tau\\)', 
                  'Pearsons \\(r\\)', 
                  'Spearmans \\(\\rho\\)')

df <- cbind('Mått' = rownames(df), df)
rownames(df) <- NULL

write.csv(df, 'inter_annotator_agreement.csv', quote = FALSE, row.names = FALSE)
