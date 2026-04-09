# Advanced R-Visualizer: The Advisor-Pleaser
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("Need gen1_path, gen2_path, out_pdf, bird_id")
gen1_path <- args[1]
gen2_path <- args[2]
out_pdf <- args[3]
bird_id <- args[4]

suppressMessages(library(pavo))

# Using a tryCatch just in case pavo hits an internal plotting error
tryCatch({
  # Use native pavo importer
  img1 <- getimg(gen1_path)
  img2 <- getimg(gen2_path)

  # Classify biological color patches into K=3 components
  class1 <- classify(img1, kcols = 3)
  class2 <- classify(img2, kcols = 3)

  # Output High-Resolution PDF
  pdf(out_pdf, width=14, height=7)
  par(mfrow=c(1,2), mar=c(2, 2, 4, 2))
  
  # Generation 1 Adjacency Network Plot
  plot(class1)
  title(main=sprintf("Bird %s: Gen 1 Colorimetry Map", bird_id), cex.main=1.5)
  
  # Generation 2 Adjacency Network Plot
  plot(class2)
  title(main=sprintf("Bird %s: Gen 2 Colorimetry Map", bird_id), cex.main=1.5)
  
  dev.off()
  cat(sprintf("R Subprocess: PDF saved to %s\n", out_pdf))
}, error = function(e) {
  cat(sprintf("R Subprocess Error: %s\n", e$message))
})
