# Installs the R packages that are not available on conda-forge / bioconda.
# Run inside the activated conda env:  Rscript scripts/install_r_packages.R
#
#   mboost, pcalg : CRAN (built from source; their dependencies come from conda)
#   SID           : CRAN
#   CAM           : archived on CRAN, installed from the archive tarball

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  Ncpus = max(1L, parallel::detectCores() - 1L)
)

missing_pkg <- function(pkg) !requireNamespace(pkg, quietly = TRUE)

for (pkg in c("mboost", "pcalg", "SID")) {
  if (missing_pkg(pkg)) install.packages(pkg)
}

if (missing_pkg("CAM")) {
  install.packages(
    "https://cloud.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz",
    repos = NULL, type = "source"
  )
}

required <- c("CAM", "SID", "mgcv", "mboost", "glmnet", "pcalg", "RBGL", "graph", "igraph")
bad <- required[vapply(required, missing_pkg, logical(1))]
if (length(bad) > 0) stop("R packages failed to install: ", paste(bad, collapse = ", "))

for (pkg in required) cat(sprintf("%-8s %s\n", pkg, as.character(packageVersion(pkg))))
cat("All R packages installed.\n")
