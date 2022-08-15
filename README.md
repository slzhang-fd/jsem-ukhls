## jsem
Paralleled full Bayesian Estimation for Multivariate Dyads Data Analysis.

### Description
The `jsem` R package we provide contains tailored estimation code for the proposed joint latent variable framework using full Bayesian methods. It uses R and C++ compiled code (Rcpp, RcppArmadillo) with OpenMP API for parallel computing to boost the estimation. Direct sampling (when applicable by using proper conjugate prior) and adaptive rejection Metropolis sampling are both employed in the program. See Appendix A in the manuscript for more details. Furthermore, practical features for MCMC sampling program such as time consumption estimation with progress bar and support for interruptions of estimation (intermediate result will be saved, even for multi-thread function) are also enabled.
