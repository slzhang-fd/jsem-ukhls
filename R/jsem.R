#' An R package for joint structural equation modeling
#' 
#' @useDynLib jsem
#' @importFrom Rcpp evalCpp
#' @importFrom stats rbinom
#' 
#' @export jsem
jsem <- function(ytp, yfp, x_covariates, b_tp_init, b_fp_init, g_init, g_free_ind, sig2_tp_init, sig2_fp_init,
                 rho_init, e_tp, e_fp, e12, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len){
  N <- nrow(ytp)
  mcmc_params_draw <- matrix(0, mcmc_len, length(g_init) + length(b_tp_init) + length(b_fp_init) + 3)
  colnames(mcmc_params_draw) <- c(paste0("g[",c(outer(1:nrow(g_init), 1:ncol(g_init), paste, sep=",")), "]"),
                                  paste0("b_tp[", 1:length(b_tp_init), "]"),
                                  paste0("b_fp[", 1:length(b_fp_init), "]"),
                                  "sig2_tp", "sig2_fp", "rho")
  mcmc_params_draw[1,] <- c(g_init, b_tp_init, b_fp_init, sig2_tp_init, sig2_fp_init, rho_init)
  time_elap <- matrix(0, mcmc_len-1, 4)
  ytp_mis_ind <- which(is.na(ytp))
  yfp_mis_ind <- which(is.na(yfp))
  for(mcmc_step in 2:mcmc_len){
    ## sample e_tp, e_fp, e12
    time1 <- Sys.time()
    e_tp <- sample_e_tp_c(e_tp, ytp, xi, b_tp_init, sig2_tp_init, beta_tp, alpha_tp, x_covariates)
    e_fp <- sample_e_fp_c(e_fp, yfp, xi, b_fp_init, sig2_fp_init, beta_fp, alpha_fp, x_covariates)
    e12 <- sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp_init, sig2_fp_init,
                        rho_init, b_tp_init, b_fp_init, x_covariates)
    ## optional: sample missing ytp and yfp
    prob_tp <- calcu_mu_tp(beta_tp, alpha_tp, x_covariates, b_tp_init, e_tp, e12[,1], xi)
    ytp[ytp_mis_ind] <- rbinom(length(ytp_mis_ind), 1, prob_tp[ytp_mis_ind])
    prob_fp <- calcu_mu_fp(beta_fp, alpha_fp, x_covariates, b_fp_init, e_fp, e12[,2], xi)
    yfp[yfp_mis_ind] <- rbinom(length(yfp_mis_ind), 1, prob_fp[yfp_mis_ind])
    ## sample g, b_tp, b_fp
    time2 <- Sys.time()
    sample_g_b(g_init, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
               b_tp_init, b_fp_init, e_tp, e_fp, e12, sig2_tp_init, sig2_fp_init, rho_init)
    #g_init <- tmp$g; b_tp_init <- tmp$b_tp; b_fp_init <- tmp$b_fp;
    ## sample sig2_tp, sig2_fp, rho
    time3 <- Sys.time()
    sig2_tp_init <- sample_sig2_tp_c(sig2_tp_init, sig2_fp_init, rho_init, e_tp, e12, xi)
    sig2_fp_init <- sample_sig2_fp_c(sig2_tp_init, sig2_fp_init, rho_init, e_fp, e12, xi)
    rho_init <- sample_rho_c(rho_init, xi, sig2_tp_init, sig2_fp_init, e12)
    ## sample xi
    time4 <- Sys.time()
    xi <- sample_xi(x_covariates, g_init, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                    ytp, b_tp_init, e_tp, yfp, b_fp_init, e_fp, e12)
    time5 <- Sys.time()
    time_elap[mcmc_step-1,] <- c(time2-time1, time3-time2, time4-time3, time5-time4)
    mcmc_params_draw[mcmc_step,] <- c(g_init, b_tp_init, b_fp_init, sig2_tp_init, sig2_fp_init, rho_init)
    cat("step: ", mcmc_step, "\t b_tp1 ", b_tp_init[1], "\t b_fp1 ", b_fp_init[1], "\t g[1,2]", g_init[1,2],
        "\t sig2_tp ", sig2_tp_init, "\t sig2_fp ", sig2_fp_init, "\t rho ", rho_init, "\n")
  }
  return(structure(list("mcmc_params_draw" = mcmc_params_draw,
              "mcmc_lv_draw_last" = list('xi'=xi, 'e_tp'=e_tp, 'e_fp'=e_fp, 'e12'=e12),
              "time_elap" = time_elap), class="jsem"))
}
#' @export jsem_update
jsem_update <- function(jsem_res, ytp, yfp, x_covariates, g_free_ind, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len){
  mcmc_len_old <- nrow(jsem_res$mcmc_params_draw)
  b_tp_old <- jsem_res$mcmc_params_draw[mcmc_len_old,grep("b_tp",colnames(jsem_res$mcmc_params_draw))]
  b_fp_old <- jsem_res$mcmc_params_draw[mcmc_len_old,grep("b_fp",colnames(jsem_res$mcmc_params_draw))]
  g_old <- matrix(jsem_res$mcmc_params_draw[mcmc_len_old,grep("g[",colnames(jsem_res$mcmc_params_draw), fixed=T)], nrow(g_free_ind))
  sig2_tp_old <- jsem_res$mcmc_params_draw[mcmc_len_old,grep("sig2_tp",colnames(jsem_res$mcmc_params_draw))]
  sig2_fp_old <- jsem_res$mcmc_params_draw[mcmc_len_old,grep("sig2_fp",colnames(jsem_res$mcmc_params_draw))]
  rho_old <- jsem_res$mcmc_params_draw[mcmc_len_old,grep("rho",colnames(jsem_res$mcmc_params_draw))]
  lv_draws_old <- jsem_res$mcmc_lv_draw_last
  jsem_res_add <- jsem(ytp, yfp, x_covariates, b_tp_old, b_fp_old, g_old, g_free_ind, sig2_tp_old, sig2_fp_old, rho_old,
                       lv_draws_old$e_tp, lv_draws_old$e_fp, lv_draws_old$e12, lv_draws_old$xi, 
                       beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len)
  # ytp, yfp, x_covariates, b_tp_init, b_fp_init, g_init, g_free_ind, sig2_tp_init, sig2_fp_init,
  # rho_init, e_tp, e_fp, e12, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len
  return(list("mcmc_params_draw" = rbind(jsem_res$mcmc_params_draw, jsem_res_add$mcmc_params_draw),
              "mcmc_lv_draw_last" = jsem_res_add$mcmc_lv_draw_last,
              "time_elap" = jsem_res_add$time_elap))
}

