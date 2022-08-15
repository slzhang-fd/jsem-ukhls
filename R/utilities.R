calcu_mu_tp <- function(beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e1, xi){
  mu_res <- matrix(0, nrow(beta_tp), ncol(beta_tp))
  mu_res[c(which(xi==1), which(xi==2)), ] <- -20
  xi_34_loc <- c(which(xi==3), which(xi==4))
  eta_tp <- as.vector(x_covariates[xi_34_loc, ] %*% b_tp + c(e_tp[which(xi==3)], e1[which(xi==4)]))
  mu_res[xi_34_loc, ] <- beta_tp[xi_34_loc,] + alpha_tp[xi_34_loc, ] * eta_tp
  return(1 / (1+exp(-mu_res)))
}
calcu_mu_fp <- function(beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e2, xi){
  mu_res <- matrix(0, nrow(beta_fp), ncol(beta_fp))
  mu_res[c(which(xi==1), which(xi==3)), ] <- -20
  xi_24_loc <- c(which(xi==2), which(xi==4))
  eta_fp <- as.vector(x_covariates[xi_24_loc, ] %*% b_fp + c(e_fp[which(xi==2)], e2[which(xi==4)]))
  mu_res[xi_24_loc, ] <- beta_fp[xi_24_loc,] + alpha_fp[xi_24_loc, ] * eta_fp
  return(1 / (1+exp(-mu_res)))
}
#' @importFrom graphics plot
#' @importFrom stats rnorm terms 
#' @importFrom utils packageVersion 
#' @importFrom coda mcmc
#' @export dylanie_plot
dylanie_plot <- function(res_j, burnin = 0){
  plot(mcmc(data=res_j[(burnin+1):length(res_j)]))
}

#' @export dylanie_summary
dylanie_summary <- function(res, burnin = 0){
  # summary(mcmc(data=res[(burnin+1):nrow(res), ]))
  g_row <- ncol(res$g_draws)/4
  g1 <- exp(res$g_draws[,1 + c(0, g_row * 1:3)])
  g1 <- g1 / rowSums(g1)
  tmp <- cbind(res$g_draws,res$b_tp_draws, res$b_fp_draws,
               res$sig2_tp_draws, res$sig2_fp_draws, res$rho_draws, g1)
  colnames(tmp) <- c(paste0("g[",c(outer(1:g_row, 1:4, paste, sep=",")), "]"),
                     paste0("b_tp[", 1:ncol(res$b_tp_draws), "]"),
                     paste0("b_fp[", 1:ncol(res$b_fp_draws), "]"),
                     "sig2_tp", "sig2_fp", "rho",
                     paste0("p[",1:4, "]"))
  summary(mcmc(data=tmp[(burnin+1):nrow(tmp), ]))
}
