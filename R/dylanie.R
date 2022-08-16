#' Paralleled full Bayesian estimation for multivariate dyads data analysis (DyLAnIE).
#' 
#' The program perform estimation for a general latent variable modelling framework.
#' This framework is intended for for the analysis of 
#' multivariate binary data on dyads, which can also handle zero inflation and 
#' non-equivalence of measurement.
#'
#' @param formula an object of class "formula" similar to linear regression. 
#' It is a symbolic description of the covariates set used in the model.
#' @param data a list for data input. It contains binary responses, covaraites 
#' matrix, a constraint matrix (1 for free parameters, 0 for fixed zero) for 
#' coefficient parameter $\phi\xi$ and pre estimated measurement parameters. See 
#' the following example for more details.
#' @param initvals a list for initial values of parameters and random variables.
#' If NULL, random initial values will be generated except for the measurement 
#' parameters.
#' @param mcmc_len an integer specifies the length of MCMC draws
#' @param verbose boolen variable, whether enable verbose mode
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{g_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\xi$.}
#'   \item{b_tp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta1$.}
#'   \item{b_fp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta2$.}
#'   \item{sig2_tp_draws}{A matrix containing MCMC draws for variance of $\eta_1$.}
#'   \item{sig2_fp_draws}{A matrix containing MCMC draws for variance of $\eta_2$.}
#'   \item{rho_draws}{A matrix containing MCMC draws for correlation of $\eta_1$ and $\eta_2$.}
#'   \item{e12_last}{A matrix containing MCMC draws for continuous random variables $\eta_1$ and $\eta_2$.}
#'   \item{xi_last}{A matrix containing MCMC draws for discrete random variables $\xi$.}
#' }
#' @references 
#' Kuha, J., Zhang, S., & Steele, F. (2021). manuscript
#' \emph{Journal of the American Statistical Association} <doi: >.
#' @examples
#' \dontrun{
#' load("jags.rda")
#' ytp <- jags.data$ytp
#' yfp <- jags.data$yfp
#' N <- nrow(ytp)
#' J <- ncol(ytp)
#' 
#' x_covariates <- matrix(0, N, 14)
#' x_covariates[,1] <- 1
#' x_covariates[,2] <- jags.data$female
#' x_covariates[,3] <- jags.data$distlong
#' x_covariates[,4] <- jags.data$age40
#' x_covariates[,5] <- jags.data$partner
#' x_covariates[,6] <- jags.data$clt2
#' x_covariates[,7] <- jags.data$c2to4
#' x_covariates[,8] <- jags.data$c5to10
#' x_covariates[,9] <- jags.data$c11to16
#' x_covariates[,10] <- jags.data$cgt16
#' x_covariates[,11] <- jags.data$jbs == 3
#' x_covariates[,12] <- jags.data$jbs == 4
#' x_covariates[,13] <- jags.data$page70
#' x_covariates[,14] <- jags.data$plive %in% c(3, 4, 5, 7, 9)
#' colnames(x_covariates) <- c('intercept', 'female', 'distlong', 'age40', 'partner', 'clt2', 'c2to4', 
#'                             'c5to10', 'c11to16', 'cgt16', 'jbs3', 'jbs4', 'page70', 'plive')
#' x_covariates <- as.data.frame(x_covariates)
#' xi <- as.vector(xi.start)
#' 
#' ## set the fixed parameters for the measurement model
#' # loadings
#' lambda_tp <- c(1.159, 1.873, 1.073, 1.560, 1.640, 1, 0.625, 0.588)
#' lambda_fp <- c(0.833, 1.293, 0.760, 0.453, 0.770, 1, 0.604, 0.578)
#' 
#' # coefficients
#' a_tp <- matrix(0, 5, 8)
#' a_fp <- matrix(0, 5, 8)
#' 
#' # intercepts
#' a_tp[1,] <- c(2.012, 1.694, -0.329, -2.335, -0.965, 0, 0.907, -1.291)
#' a_fp[1,] <- c(1.855, 2.621, 2.107, 2.473, 0.488, 0, 0.514, 0.991)
#' 
#' # female
#' a_tp[2,] <- c(-0.635, 0, 0, 0, 0, 0, -1.529, -0.642)
#' a_fp[2,] <- c(0, 0, -0.222, 0, 0, 0, 0, -0.322)
#' 
#' # distlong
#' a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.038, 0)
#' a_fp[3,] <- c(1.100, 0.603, 1.997, 0.689, 2.536, 0, -0.685, 0)
#' 
#' # etaXfemale
#' a_tp[4,] <- c(-0.461, 0, 0, 0, 0, 0, 0.090, -0.030)
#' a_fp[4,] <- c(0, 0, 0.054, 0, 0, 0, 0, -0.057)
#' 
#' # etaXdistlong
#' a_tp[5,] <- c(0.905, 1.225, 0.952, 0, 0.626, 0, 0.535, 0)
#' a_fp[5,] <- c(0.898, 0.740, 1.039, 0.618, 1.213, 0, 0.049, 0)
#' 
#' beta_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_tp[1:3,]
#' beta_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% a_fp[1:3,]
#' 
#' alpha_tp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_tp, a_tp[4:5,])
#' alpha_fp <- cbind(rep(1, N), jags.data$female, jags.data$distlong) %*% rbind(lambda_fp, a_fp[4:5,])
#' 
#' g_free_ind <- matrix(1, 14, 4)
#' g_free_ind[,1] <- g_free_ind[3,2] <- g_free_ind[6,3] <- g_free_ind[7,3] <- 0
#' 
#' aidrp_data <- list(ytp=ytp, yfp = yfp, x_covariates = x_covariates,
#'                    g_free_ind = g_free_ind, beta_tp = beta_tp, beta_fp = beta_fp,
#'                    alpha_tp = alpha_tp, alpha_fp = alpha_fp, xi=xi)
#' 
#' dylanie_res <- dylanie_model(~female, data=aidrp_data, mcmc_len = 50, verbose = F)
#' }
#' @export dylanie_model
dylanie_model <- function(formula, data, initvals=NULL, mcmc_len, verbose=F){
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  if(is.null(initvals)){
    N <- nrow(data$ytp)
    g_init <- matrix(0, 14, 4)
    g_init[,1] <- 0
    g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
    initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
                     g_init = g_init, sig2_tp_init = 1/0.35,
                     sig2_fp_init = 1/0.2, rho_init = 0,
                     e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2), xi = data$xi)
  }
  res_cpp <- jsem_cpp(data$ytp, data$yfp, as.matrix(data$x_covariates[c('intercept',term_label)]), 
                      matrix(initvals$b_tp_init[1:nparam],ncol=1), matrix(initvals$b_fp_init[1:nparam],ncol=1), 
                      matrix(initvals$g_init[1:nparam,],nrow=nparam), matrix(data$g_free_ind[1:nparam,],nrow=nparam), 
                      initvals$sig2_tp_init, initvals$sig2_fp_init, initvals$rho_init,
                      initvals$e_tp, initvals$e_fp, initvals$e12, initvals$xi, 
                      data$beta_tp, data$alpha_tp, data$beta_fp, data$alpha_fp, mcmc_len, verbose)
  return(res_cpp)
}


#' Paralleled full Bayesian estimation for multivariate dyads data analysis (DyLAnIE).
#' 
#' The program perform estimation for a general latent variable modelling framework.
#' This framework is intended for for the analysis of 
#' multivariate binary data on dyads, which can also handle zero inflation and 
#' non-equivalence of measurement.
#'
#' @param formula an object of class "formula" similar to linear regression. 
#' It is a symbolic description of the covariates set used in the model.
#' @param data a list for data input. It contains binary responses, covaraites 
#' matrix, a constraint matrix (1 for free parameters, 0 for fixed zero) for 
#' coefficient parameter $\phi\xi$ and pre estimated measurement parameters. See 
#' the following example for more details.
#' @param mcmc_len an integer specifies the length of MCMC draws
#' @param verbose boolen variable, whether enable verbose mode
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{g_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\xi$.}
#'   \item{b_tp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta1$.}
#'   \item{b_fp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta2$.}
#'   \item{sig2_tp_draws}{A matrix containing MCMC draws for variance of $\eta_1$.}
#'   \item{sig2_fp_draws}{A matrix containing MCMC draws for variance of $\eta_2$.}
#'   \item{rho_draws}{A matrix containing MCMC draws for correlation of $\eta_1$ and $\eta_2$.}
#'   \item{e12_last}{A matrix containing MCMC draws for continuous random variables $\eta_1$ and $\eta_2$.}
#'   \item{xi_last}{A matrix containing MCMC draws for discrete random variables $\xi$.}
#' }
#' @references 
#' Kuha, J., Zhang, S., & Steele, F. (2021). manuscript
#' \emph{Journal of the American Statistical Association} <doi: >.
#' @examples
#' \dontrun{
#' # load the simulated dataset
#' load("jags.rda")
#' aidrp_data_simple <- list(ytp=jags.data$ytp, yfp = jags.data$yfp, xi=xi.start,
#'                           female = jags.data$female,
#'                           distlong = jags.data$distlong,
#'                           age40 = jags.data$age40,
#'                           partner = jags.data$partner,
#'                           clt2 = jags.data$clt2,
#'                           c2to4 = jags.data$c2to4,
#'                           c5to10 = jags.data$c5to10,
#'                           c11to16 = jags.data$c11to16,
#'                           cgt16 = jags.data$cgt16,
#'                           jbs3 = jags.data$jbs == 3,
#'                           jbs4 = jags.data$jbs == 4,
#'                           page70 = jags.data$page70,
#'                           plive = jags.data$plive %in% c(3, 4, 5, 7, 9))
#' 
#' dylanie_res <- dylanie_model_simple(~female+age40, data=aidrp_data_simple, mcmc_len = 10, verbose = F)
#' }
#' 
#' @export dylanie_model_simple
dylanie_model_simple <- function(formula, data, mcmc_len, verbose = F){
  N <- nrow(data$ytp)
  J <- ncol(data$ytp)
  
  ## set the fixed parameters for the measurement model
  # loadings
  lambda_tp <- c(1.159, 1.873, 1.073, 1.560, 1.640, 1, 0.625, 0.588)
  lambda_fp <- c(0.833, 1.293, 0.760, 0.453, 0.770, 1, 0.604, 0.578)
  
  # coefficients
  a_tp <- matrix(0, 5, 8)
  a_fp <- matrix(0, 5, 8)
  
  # intercepts
  a_tp[1,] <- c(2.012, 1.694, -0.329, -2.335, -0.965, 0, 0.907, -1.291)
  a_fp[1,] <- c(1.855, 2.621, 2.107, 2.473, 0.488, 0, 0.514, 0.991)
  
  # female
  a_tp[2,] <- c(-0.635, 0, 0, 0, 0, 0, -1.529, -0.642)
  a_fp[2,] <- c(0, 0, -0.222, 0, 0, 0, 0, -0.322)
  
  # distlong
  # a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.038, 0)
  a_tp[3,] <- c(-0.831, -0.128, 0.415, 0, 0.466, 0, 0.056, 0)
  a_fp[3,] <- c(1.100, 0.603, 1.997, 0.689, 2.536, 0, -0.685, 0)
  
  # etaXfemale
  a_tp[4,] <- c(-0.461, 0, 0, 0, 0, 0, 0.090, -0.030)
  a_fp[4,] <- c(0, 0, 0.054, 0, 0, 0, 0, -0.057)
  
  # etaXdistlong
  a_tp[5,] <- c(0.905, 1.225, 0.952, 0, 0.626, 0, 0.535, 0)
  a_fp[5,] <- c(0.898, 0.740, 1.039, 0.618, 1.213, 0, 0.049, 0)
  
  beta_tp <- cbind(rep(1, N), data$female, data$distlong) %*% a_tp[1:3,]
  beta_fp <- cbind(rep(1, N), data$female, data$distlong) %*% a_fp[1:3,]
  
  alpha_tp <- cbind(rep(1, N), data$female, data$distlong) %*% rbind(lambda_tp, a_tp[4:5,])
  alpha_fp <- cbind(rep(1, N), data$female, data$distlong) %*% rbind(lambda_fp, a_fp[4:5,])
  
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  
  g_free_ind <- matrix(1, nparam, 4)
  # g_free_ind[,1] <- g_free_ind[3,2] <- g_free_ind[6,3] <- g_free_ind[7,3] <- 0
  
  g_init <- matrix(0, nparam, 4)
  # g_init[,1] <- 0
  # g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
  initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
                   g_init = g_init, sig2_tp_init = 1/0.35,
                   sig2_fp_init = 1/0.2, rho_init = 0,
                   e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2))
  x_covariates <- matrix(rep(1, N), ncol = 1)
  if(length(term_label)) x_covariates <- cbind(x_covariates, as.matrix(data.frame(data[term_label])))
  res_cpp <- jsem_cpp(data$ytp, data$yfp, x_covariates, 
                      initvals$b_tp_init, initvals$b_fp_init, 
                      matrix(g_init[1:nparam,],nrow=nparam), matrix(g_free_ind[1:nparam,],nrow=nparam), 
                      initvals$sig2_tp_init, initvals$sig2_fp_init, initvals$rho_init,
                      initvals$e_tp, initvals$e_fp, initvals$e12, data$xi, 
                      beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len, verbose)
  res_cpp <- c(res_cpp, 
               list("ytp"= data$ytp, 
                    "yfp"=data$yfp, 
                    "x_covariates"=x_covariates,
                    "g_free_ind"=g_free_ind,
                    "beta_tp"=beta_tp,
                    "alpha_tp"=alpha_tp,
                    "beta_fp"=beta_fp,
                    "alpha_fp"=alpha_fp))
  return(res_cpp)
}

#' Paralleled full Bayesian estimation for multivariate dyads data analysis (DyLAnIE).
#' 
#' The program perform estimation for a general latent variable modelling framework.
#' This framework is intended for for the analysis of 
#' multivariate binary data on dyads, which can also handle zero inflation and 
#' non-equivalence of measurement.
#'
#' @param dylanie_res object return from `dylanie_model()` or `dylanie_model_simple()` function
#' @param mcmc_len an integer specifies the length of MCMC draws
#' @param verbose boolen variable, whether enable verbose mode
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{g_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\xi$.}
#'   \item{b_tp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta1$.}
#'   \item{b_fp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta2$.}
#'   \item{sig2_tp_draws}{A matrix containing MCMC draws for variance of $\eta_1$.}
#'   \item{sig2_fp_draws}{A matrix containing MCMC draws for variance of $\eta_2$.}
#'   \item{rho_draws}{A matrix containing MCMC draws for correlation of $\eta_1$ and $\eta_2$.}
#'   \item{e12_last}{A matrix containing MCMC draws for continuous random variables $\eta_1$ and $\eta_2$.}
#'   \item{xi_last}{A matrix containing MCMC draws for discrete random variables $\xi$.}
#' }
#' @references 
#' Kuha, J., Zhang, S., & Steele, F. (2021). manuscript
#' \emph{Journal of the American Statistical Association} <doi: >.
#' @examples
#' \dontrun{
#' dylanie_res_updated <- dylanie_model_update(dylanie_res, 10, verbose = F)
#' }
#' @export dylanie_model_update
dylanie_model_update <- function(dylanie_res, mcmc_len, verbose = F){
  mcmc_len_old <- nrow(dylanie_res$g_draws)
  b_tp_old <- dylanie_res$b_tp_draws[mcmc_len_old,]
  b_fp_old <- dylanie_res$b_fp_draws[mcmc_len_old,]
  g_old <- matrix(dylanie_res$g_draws[mcmc_len_old,], ncol = 4)
  updated_res <- jsem_cpp(dylanie_res$ytp, dylanie_res$yfp, dylanie_res$x_covariates, 
                          matrix(b_tp_old, ncol = 1), matrix(b_fp_old, ncol = 1), 
                          g_old, dylanie_res$g_free_ind, 
                          dylanie_res$sig2_tp_draws[mcmc_len_old], dylanie_res$sig2_fp_draws[mcmc_len_old], 
                          dylanie_res$rho_draws[mcmc_len_old],
                          dylanie_res$e_tp_last, dylanie_res$e_fp_last, dylanie_res$e12_last, dylanie_res$xi_last, 
                          dylanie_res$beta_tp, dylanie_res$alpha_tp, dylanie_res$beta_fp, dylanie_res$alpha_fp, 
                          mcmc_len, verbose)
  return(list("g_draws" = rbind(dylanie_res$g_draws, updated_res$g_draws),
              "b_tp_draws" = rbind(dylanie_res$b_tp_draws, updated_res$b_tp_draws),
              "b_fp_draws" = rbind(dylanie_res$b_fp_draws, updated_res$b_fp_draws),
              "sig2_tp_draws" = c(dylanie_res$sig2_tp_draws, updated_res$sig2_tp_draws),
              "sig2_fp_draws" = c(dylanie_res$sig2_fp_draws, updated_res$sig2_fp_draws),
              "rho_draws" = c(dylanie_res$rho_draws, updated_res$rho_draws),
              "e_tp_last" = updated_res$e_tp_last,
              "e_fp_last" = updated_res$e_fp_last,
              "e12_last" = updated_res$e12_last,
              "xi_last" = updated_res$xi_last,
              "ytp" = dylanie_res$ytp,
              "yfp" = dylanie_res$yfp,
              "x_covariates" = dylanie_res$x_covariates,
              "g_free_ind" = dylanie_res$g_free_ind,
              "beta_tp" = dylanie_res$beta_tp,
              "alpha_tp" = dylanie_res$alpha_tp,
              "beta_fp" = dylanie_res$beta_fp,
              "alpha_fp" = dylanie_res$alpha_fp))
}

#' Paralleled full Bayesian estimation for multivariate dyads data analysis (DyLAnIE).
#' 
#' The program perform estimation for a general latent variable modelling framework.
#' This framework is intended for for the analysis of 
#' multivariate binary data on dyads, which can also handle zero inflation and 
#' non-equivalence of measurement.
#'
#' @param formula an object of class "formula" similar to linear regression. 
#' It is a symbolic description of the covariates set used in the model.
#' @param data a list for data input. It contains binary responses, covaraites 
#' matrix, a constraint matrix (1 for free parameters, 0 for fixed zero) for 
#' coefficient parameter $\phi\xi$ and pre estimated measurement parameters. See 
#' the following example for more details.
#' @param mcmc_len an integer specifies the length of MCMC draws
#' @param chains an integer specifies the number of MCMC chains (draw at the same time)
#' @param verbose boolen variable, whether enable verbose mode
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{g_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\xi$.}
#'   \item{b_tp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta1$.}
#'   \item{b_fp_draws}{A matrix containing MCMC draws for coefficient parameters $\phi_\eta2$.}
#'   \item{sig2_tp_draws}{A matrix containing MCMC draws for variance of $\eta_1$.}
#'   \item{sig2_fp_draws}{A matrix containing MCMC draws for variance of $\eta_2$.}
#'   \item{rho_draws}{A matrix containing MCMC draws for correlation of $\eta_1$ and $\eta_2$.}
#'   \item{e12_last}{A matrix containing MCMC draws for continuous random variables $\eta_1$ and $\eta_2$.}
#'   \item{xi_last}{A matrix containing MCMC draws for discrete random variables $\xi$.}
#' }
#' @references 
#' Kuha, J., Zhang, S., & Steele, F. (2021). manuscript
#' \emph{Journal of the American Statistical Association} <doi: >.
#' @examples
#' \dontrun{
#' dylanie_mchains_res <- dylanie_model_mchains(~female, data=aidrp_data, mcmc_len = 10, chains = 2, verbose = F)
#' str(dylanie_mchains_res)
#' }
#' 
#' @export dylanie_model_mchains
dylanie_model_mchains <- function(formula, data, mcmc_len, chains=1,verbose=F){
  term_label <- unlist(attributes(terms(formula))['term.labels'])
  nparam <- length(term_label) + 1
  N <- nrow(data$ytp)
  g_init <- matrix(0, 14, 4)
  g_init[,1] <- 0
  g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
  b_tp_init <- rep(0, 14)
  b_fp_init <- rep(0, 14)
  res_cpp <- jsem_multi_chains(data$ytp, data$yfp, as.matrix(data$x_covariates[c('intercept',term_label)]), 
                               matrix(b_tp_init[1:nparam],ncol=1), matrix(b_fp_init[1:nparam],ncol=1), 
                               matrix(g_init[1:nparam,],nrow=nparam), matrix(data$g_free_ind[1:nparam,],nrow=nparam), 
                               1/0.35, 1/0.2, 0, data$xi,
                               data$beta_tp, data$alpha_tp, data$beta_fp, data$alpha_fp, mcmc_len, chains)
  return(res_cpp)
}