// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "arms_ori.h"
#include "jsem_omp.h"
#include "progress.hpp"
#include "eta_progress_bar.hpp"

arma::mat sample_g_c(arma::mat g, const arma::mat &g_free_ind, const arma::uvec &xi, 
                     const arma::mat &x_covariates);
arma::vec sample_e_tp_c(arma::vec e_tp, arma::mat ytp, arma::uvec xi, arma::vec b_tp,
                        double sig2_tp, arma::mat beta_tp, arma::mat alpha_tp, arma::mat x_covariates);
arma::vec sample_e_fp_c(arma::vec e_fp, arma::mat yfp, arma::uvec xi, arma::vec b_fp,
                        double sig2_fp, arma::mat beta_fp, arma::mat alpha_fp, arma::mat x_covariates);
arma::mat sample_e12_c(arma::mat e12, arma::mat ytp, arma::mat yfp, arma::uvec xi,
                       arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                       double sig2_tp, double sig2_fp, double rho,
                       arma::vec b_tp, arma::vec b_fp, arma::mat x_covariates);
arma::vec sample_b_tp_c(arma::vec b_tp, const arma::mat &ytp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_tp, 
                        const arma::mat &alpha_tp, const arma::vec &e_tp, const arma::vec &e1);
arma::vec sample_b_fp_c(arma::vec b_fp, const arma::mat &yfp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_fp, 
                        const arma::mat &alpha_fp, const arma::vec &e_fp, const arma::vec &e2);
arma::vec sample_b_linear_reg(arma::vec e_tp1, arma::vec e_tp2, arma::vec e_fp2,
                              arma::mat x_covariates1, arma::mat x_covariates2,
                              arma::vec b_tp, arma::vec b_fp, 
                              double sig2_tp, double sig2_fp, double rho);
arma::vec sample_b_linear_reg_new(arma::vec e1, arma::vec e2,
                                  const arma::mat &x_covariates,
                                  arma::vec b_tp, arma::vec b_fp, 
                                  double sig2_tp, double sig2_fp, double rho);

arma::mat sample_e12_c(arma::mat e12, arma::mat ytp, arma::mat yfp, arma::vec xi,
                       arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                       double sig2_tp, double sig2_fp, double rho,
                       arma::vec b_tp, arma::vec b_fp, arma::mat x_covariates);
void sample_e12_new(arma::mat &e12, const arma::mat &ytp, const arma::mat &yfp, const arma::uvec &xi,
                    const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                    const arma::mat &beta_fp, const arma::mat &alpha_fp, 
                    double sig2_tp, double sig2_fp, double rho,
                    const arma::vec &b_tp, const arma::vec &b_fp, 
                    const arma::mat &x_covariates);
double sample_sig2_tp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_tp, arma::mat e12, arma::uvec xi);
double sample_sig2_fp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_fp, arma::mat e12, arma::uvec xi);
double sample_rho_c(double rho, const arma::uvec &xi, double sig2_tp, double sig2_fp, const arma::mat &e12);
arma::mat sample_Sigma_new(const arma::mat &e12);

arma::uvec sample_xi(arma::mat x_covariates, arma::mat g,
                    arma::mat beta_tp, arma::mat alpha_tp, arma::mat beta_fp, arma::mat alpha_fp,
                    arma::mat ytp, arma::vec b_tp, arma::vec e_tp,
                    arma::mat yfp, arma::vec b_fp, arma::vec e_fp,
                    arma::mat e12);
arma::uvec sample_xi_new(const arma::mat &x_covariates, const arma::mat &g,
                         const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                         const arma::mat &beta_fp, const arma::mat &alpha_fp,
                         const arma::mat &ytp, const arma::vec &b_tp,
                         const arma::mat &yfp, const arma::vec &b_fp,
                         const arma::mat &e12);
arma::uvec sample_xi_new1(const arma::mat &x_covariates, const arma::mat &g,
                         const arma::mat &beta_tp, const arma::mat &alpha_tp, 
                         const arma::mat &beta_fp, const arma::mat &alpha_fp,
                         const arma::mat &ytp, const arma::vec &b_tp,
                         const arma::mat &yfp, const arma::vec &b_fp,
                         const arma::mat &e12);
void impute_ytp_mis(arma::mat &ytp, arma::uvec ytp_mis_ind, arma::mat beta_tp, arma::mat alpha_tp,
                    arma::mat x_covariates, arma::vec b_tp, arma::vec e_tp, arma::vec e1, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_tp));
  arma::uvec xi_3_loc = arma::find(xi==3);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_34_loc = arma::join_cols(xi_3_loc, xi_4_loc);
  arma::vec eta_tp = x_covariates.rows(xi_34_loc) * b_tp + 
    arma::join_cols(e_tp(xi_3_loc), e1(xi_4_loc));
  arma::mat tmp = alpha_tp.rows(xi_34_loc);
  tmp.each_col() %= eta_tp;
  mu_res.rows(xi_34_loc) = beta_tp.rows(xi_34_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  ytp(ytp_mis_ind) = draws(ytp_mis_ind);
  if ( Progress::check_abort() )
    return;
}
void impute_ytp_mis_core(arma::mat &ytp, arma::uvec ytp_mis_ind, arma::mat beta_tp, arma::mat alpha_tp,
                    arma::mat x_covariates, arma::vec b_tp, arma::vec e_tp, arma::vec e1, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_tp));
  arma::uvec xi_3_loc = arma::find(xi==3);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_34_loc = arma::join_cols(xi_3_loc, xi_4_loc);
  arma::vec eta_tp = x_covariates.rows(xi_34_loc) * b_tp + 
    arma::join_cols(e_tp(xi_3_loc), e1(xi_4_loc));
  arma::mat tmp = alpha_tp.rows(xi_34_loc);
  tmp.each_col() %= eta_tp;
  mu_res.rows(xi_34_loc) = beta_tp.rows(xi_34_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  ytp(ytp_mis_ind) = draws(ytp_mis_ind);
}
void impute_yfp_mis(arma::mat &yfp, arma::uvec yfp_mis_ind, arma::mat beta_fp, arma::mat alpha_fp,
                    arma::mat x_covariates, arma::vec b_fp, arma::vec e_fp, arma::vec e2, arma::uvec xi){
  arma::mat mu_res = -20 * arma::ones(arma::size(beta_fp));
  arma::uvec xi_2_loc = arma::find(xi==2);
  arma::uvec xi_4_loc = arma::find(xi==4);
  arma::uvec xi_24_loc = arma::join_cols(xi_2_loc, xi_4_loc);
  arma::vec eta_fp = x_covariates.rows(xi_24_loc) * b_fp + 
    arma::join_cols(e_fp(xi_2_loc), e2(xi_4_loc));
  arma::mat tmp = alpha_fp.rows(xi_24_loc);
  tmp.each_col() %= eta_fp;
  mu_res.rows(xi_24_loc) = beta_fp.rows(xi_24_loc) + tmp;
  arma::mat prob = 1.0 / (1.0 + arma::exp(-mu_res));
  arma::mat temp(arma::size(prob),arma::fill::randu);
  arma::mat draws = arma::conv_to<arma::mat>::from(temp<prob);
  yfp(yfp_mis_ind) = draws(yfp_mis_ind);
}
// [[Rcpp::export]]
void sample_g_b(arma::mat &g, const arma::mat &g_free_ind, const arma::uvec &xi, 
                const arma::mat &x_covariates, const arma::mat &ytp, const arma::mat &yfp,
                const arma::mat &beta_tp, const arma::mat &alpha_tp,
                const arma::mat &beta_fp, const arma::mat &alpha_fp,
                arma::vec &b_tp, arma::vec &b_fp,
                const arma::vec &e_tp, const arma::vec &e_fp,
                const arma::mat &e12, const double sig2_tp, const double sig2_fp, const double rho){
  arma::mat g_res;
  arma::vec b_tp_res,b_fp_res;
  arma::vec e1 = e12.col(0);
  arma::vec e2 = e12.col(1);
  #pragma omp parallel sections num_threads(getjsem_threads())
  {
    #pragma omp section
    {
      g_res = sample_g_c(g, g_free_ind, xi, x_covariates);
    }
    #pragma omp section
    {
      // b_tp_res = sample_b_tp_c(b_tp, ytp, xi, x_covariates, beta_tp, alpha_tp, e_tp, e1);
      arma::uvec xi_3_loc = arma::find(xi==3);
      arma::uvec xi_4_loc = arma::find(xi==4);
      b_tp_res = sample_b_linear_reg(e_tp(xi_3_loc), e1(xi_4_loc), e2(xi_4_loc),
                                    x_covariates.rows(xi_3_loc), x_covariates.rows(xi_4_loc), 
                                    b_tp, b_fp, sig2_tp, sig2_fp, rho);
    }
    #pragma omp section
    {
      // b_fp_res = sample_b_fp_c(b_fp, yfp, xi, x_covariates, beta_fp, alpha_fp, e_fp, e2);
      arma::uvec xi_2_loc = arma::find(xi==2);
      arma::uvec xi_4_loc = arma::find(xi==4);
      b_fp_res = sample_b_linear_reg(e_fp(xi_2_loc), e2(xi_4_loc), e1(xi_4_loc),
                                     x_covariates.rows(xi_2_loc), x_covariates.rows(xi_4_loc),
                                     b_fp, b_tp, sig2_fp, sig2_tp, rho);
    }
  }
  g = g_res;
  b_tp = b_tp_res;
  b_fp = b_fp_res;
}
//' @export
// [[Rcpp::export]]
Rcpp::List jsem_cpp(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, 
                    arma::vec e_tp, arma::vec e_fp, arma::mat e12, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::vec sig2_tp_draws = arma::zeros(mcmc_len);
  arma::vec sig2_fp_draws = arma::zeros(mcmc_len);
  arma::vec rho_draws = arma::zeros(mcmc_len);
  
  arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);
  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);
  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
    if ( p.increment() ) {
      // // sample missing ytp and yfp
      impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e12.col(0), xi);
      impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e12.col(1), xi);

      e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      e12 = sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp,
                         rho, b_tp, b_fp, x_covariates);
      
      // sample g, b_tp, b_fp
      sample_g_b(g, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
                 b_tp, b_fp, e_tp, e_fp, e12, sig2_tp, sig2_fp, rho);
      // sample sig2_tp, sig2_fp, rho
      sig2_tp = sample_sig2_tp_c(sig2_tp, sig2_fp, rho, e_tp, e12, xi);
      sig2_fp = sample_sig2_fp_c(sig2_tp, sig2_fp, rho, e_fp, e12, xi);
      rho = sample_rho_c(rho, xi, sig2_tp, sig2_fp, e12);
      // sample xi
      xi = sample_xi(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                     ytp, b_tp, e_tp, yfp, b_fp, e_fp, e12);
      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      sig2_tp_draws(i) = sig2_tp;
      sig2_fp_draws(i) = sig2_fp;
      rho_draws(i) = rho;
      if(verbose)
        Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho %f\n",
                i, b_tp(0), b_fp(0), g(0,1), sig2_tp, sig2_fp, rho);
    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                             Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                             Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                             Rcpp::Named("sig2_tp_draws") = sig2_tp_draws,
                             Rcpp::Named("sig2_fp_draws") = sig2_fp_draws,
                             Rcpp::Named("rho_draws") = rho_draws,
                             Rcpp::Named("e_tp_last") = e_tp,
                             Rcpp::Named("e_fp_last") = e_fp,
                             Rcpp::Named("e12_last") = e12,
                             Rcpp::Named("xi_last") = xi);
                             // Rcpp::Named("ytp") = ytp,
                             // Rcpp::Named("yfp") = yfp,
                             // Rcpp::Named("x_covariates") = x_covariates,
                             // Rcpp::Named("g_free_ind") = g_free_ind,
                             // Rcpp::Named("beta_tp") = beta_tp,
                             // Rcpp::Named("alpha_tp") = alpha_tp,
                             // Rcpp::Named("beta_fp") = beta_fp,
                             // Rcpp::Named("alpha_fp") = alpha_fp);
}
//' @export
// [[Rcpp::export]]
Rcpp::List jsem_cpp1(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, 
                    arma::mat e12, arma::uvec xi, arma::mat beta_tp, arma::mat alpha_tp,
                    arma::mat beta_fp, arma::mat alpha_fp, int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::vec sig2_tp_draws = arma::zeros(mcmc_len);
  arma::vec sig2_fp_draws = arma::zeros(mcmc_len);
  arma::vec rho_draws = arma::zeros(mcmc_len);
  
  // arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  // arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);
  ETAProgressBar pb;
  Progress p(mcmc_len, !verbose, pb);
  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
    if ( p.increment() ) {
      // sampling of 'missing' responses are depracated due to that they are actually inapplicable
      // // sample missing ytp and yfp
      // impute_ytp_mis(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e12.col(0), xi);
      // impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e12.col(1), xi);
      
      // sampling of two versions of latent continuous variable e or eta is depracated
      // now a only one version of latent continuous variables (help tendencies) are used
      
      // e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      // e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      // e12 = sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp,
      //                    rho, b_tp, b_fp, x_covariates);
      
      // sample e or eta new, e12 is replaced in place
      // e12(:,0) is e_tp, e12(:,1) is e_fp
      sample_e12_new(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp, rho,
                           b_tp, b_fp, x_covariates);
      
      // Rprintf("sample e12 finished | ");
      // sample g, b_tp, b_fp
      // sample_g_b(g, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
      //            b_tp, b_fp, e_tp, e_fp, e12, sig2_tp, sig2_fp, rho)
      
      arma::vec e1 = e12.col(0);
      arma::vec e2 = e12.col(1);
      #pragma omp parallel sections num_threads(getjsem_threads())
      {
          #pragma omp section
          {
            g = sample_g_c(g, g_free_ind, xi, x_covariates);
          }
          #pragma omp section
          {
            b_tp = sample_b_linear_reg_new(e12.col(0), e12.col(1), x_covariates,
                                           b_tp, b_fp, sig2_tp, sig2_fp, rho);
          }
          #pragma omp section
          {
            b_fp = sample_b_linear_reg_new(e12.col(1), e12.col(0), x_covariates,
                                           b_fp, b_tp, sig2_fp, sig2_tp, rho);
          }
      }
      // Rprintf("sample g b finished | ");
      // sample sig2_tp, sig2_fp, rho
      // sig2_tp = sample_sig2_tp_c(sig2_tp, sig2_fp, rho, e_tp, e12, xi);
      // sig2_fp = sample_sig2_fp_c(sig2_tp, sig2_fp, rho, e_fp, e12, xi);
      // rho = sample_rho_c(rho, xi, sig2_tp, sig2_fp, e12);
      // previous sampling of covariance matrix is depracated
      
      // sample Sigma (covariance matrix for eta)
      arma::mat Sigma = sample_Sigma_new(e12);
      sig2_tp = Sigma(0,0);
      sig2_fp = Sigma(1,1);
      rho = Sigma(0,1) / std::sqrt(sig2_tp * sig2_fp);
      // Rprintf("sample Sigma finished | ");
      
      // sample xi
      xi = sample_xi_new1(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                     ytp, b_tp, yfp, b_fp, e12);
      // Rprintf("sample xi finished | ");
      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      sig2_tp_draws(i) = sig2_tp;
      sig2_fp_draws(i) = sig2_fp;
      rho_draws(i) = rho;
      if(verbose)
        Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho %f\n",
                i, b_tp(0), b_fp(0), g(0,1), sig2_tp, sig2_fp, rho);
    }
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("sig2_tp_draws") = sig2_tp_draws,
                            Rcpp::Named("sig2_fp_draws") = sig2_fp_draws,
                            Rcpp::Named("rho_draws") = rho_draws,
                            Rcpp::Named("e12_last") = e12,
                            Rcpp::Named("xi_last") = xi,
                            Rcpp::Named("ytp") = ytp,
                            Rcpp::Named("yfp") = yfp,
                            Rcpp::Named("x_covariates") = x_covariates,
                            Rcpp::Named("g_free_ind") = g_free_ind,
                            Rcpp::Named("beta_tp") = beta_tp,
                            Rcpp::Named("alpha_tp") = alpha_tp,
                            Rcpp::Named("beta_fp") = beta_fp,
                            Rcpp::Named("alpha_fp") = alpha_fp);
}
Rcpp::List jsem_core(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, 
                    arma::vec e_tp, arma::vec e_fp, arma::mat e12, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, bool verbose=false){
  arma::mat g_draws = arma::zeros(g.n_elem,mcmc_len);
  arma::mat b_tp_draws = arma::zeros(b_tp.n_elem,mcmc_len);
  arma::mat b_fp_draws = arma::zeros(b_fp.n_elem,mcmc_len);
  arma::vec sig2_tp_draws = arma::zeros(mcmc_len);
  arma::vec sig2_fp_draws = arma::zeros(mcmc_len);
  arma::vec rho_draws = arma::zeros(mcmc_len);
  
  arma::uvec ytp_mis_ind = arma::find_nonfinite(ytp);
  arma::uvec yfp_mis_ind = arma::find_nonfinite(yfp);
  for(int i=0;i<mcmc_len;++i){
    // sample e_tp, e_fp, e12
      e_tp = sample_e_tp_c(e_tp, ytp, xi, b_tp, sig2_tp, beta_tp, alpha_tp, x_covariates);
      e_fp = sample_e_fp_c(e_fp, yfp, xi, b_fp, sig2_fp, beta_fp, alpha_fp, x_covariates);
      e12 = sample_e12_c(e12, ytp, yfp, xi, beta_tp, alpha_tp, beta_fp, alpha_fp, sig2_tp, sig2_fp,
                         rho, b_tp, b_fp, x_covariates);
      // sample missing ytp and yfp
      impute_ytp_mis_core(ytp, ytp_mis_ind, beta_tp, alpha_tp, x_covariates, b_tp, e_tp, e12.col(0), xi);
      impute_yfp_mis(yfp, yfp_mis_ind, beta_fp, alpha_fp, x_covariates, b_fp, e_fp, e12.col(1), xi);
      // sample g, b_tp, b_fp
      sample_g_b(g, g_free_ind, xi, x_covariates, ytp, yfp, beta_tp, alpha_tp, beta_fp, alpha_fp,
                 b_tp, b_fp, e_tp, e_fp, e12, sig2_tp, sig2_fp, rho);
      // sample sig2_tp, sig2_fp, rho
      sig2_tp = sample_sig2_tp_c(sig2_tp, sig2_fp, rho, e_tp, e12, xi);
      sig2_fp = sample_sig2_fp_c(sig2_tp, sig2_fp, rho, e_fp, e12, xi);
      rho = sample_rho_c(rho, xi, sig2_tp, sig2_fp, e12);
      // sample xi
      xi = sample_xi(x_covariates, g, beta_tp, alpha_tp, beta_fp, alpha_fp, 
                     ytp, b_tp, e_tp, yfp, b_fp, e_fp, e12);
      g_draws.col(i) = g.as_col();
      b_tp_draws.col(i) = b_tp;
      b_fp_draws.col(i) = b_fp;
      sig2_tp_draws(i) = sig2_tp;
      sig2_fp_draws(i) = sig2_fp;
      rho_draws(i) = rho;
      if(verbose)
        Rprintf("step: %d\t b_tp1 %f\t b_fp1 %f\t g[1,2] %f\t sig2_tp %f\t sig2_fp %f\t rho %f\n",
                i, b_tp(0), b_fp(0), g(0,1), sig2_tp, sig2_fp, rho);
  }
  return Rcpp::List::create(Rcpp::Named("g_draws") = g_draws.t(),
                            Rcpp::Named("b_tp_draws") = b_tp_draws.t(),
                            Rcpp::Named("b_fp_draws") = b_fp_draws.t(),
                            Rcpp::Named("sig2_tp_draws") = sig2_tp_draws,
                            Rcpp::Named("sig2_fp_draws") = sig2_fp_draws,
                            Rcpp::Named("rho_draws") = rho_draws,
                            Rcpp::Named("e_tp_last") = e_tp,
                            Rcpp::Named("e_fp_last") = e_fp,
                            Rcpp::Named("e12_last") = e12,
                            Rcpp::Named("xi_last") = xi);
}
// [[Rcpp::export]]
Rcpp::List jsem_multi_chains(arma::mat ytp, arma::mat yfp, arma::mat x_covariates, arma::vec b_tp, arma::vec b_fp,
                    arma::mat g, arma::mat g_free_ind, double sig2_tp, double sig2_fp, double rho, arma::uvec xi, 
                    arma::mat beta_tp, arma::mat alpha_tp,arma::mat beta_fp, arma::mat alpha_fp,
                    int mcmc_len, int chains = 1){
  Rcpp::List res(chains);
  int old_thread_num = setjsem_threads(1);
  int N = ytp.n_rows;
#pragma omp parallel num_threads(std::min(chains, omp_get_num_procs()))
{
  if(omp_get_thread_num() == 0){
    // N <- nrow(data$ytp)
    // g_init <- matrix(0, 14, 4)
    // g_init[,1] <- 0
    // g_init[3,2] <- g_init[6,3] <- g_init[7,3] <- -5
    // initvals <- list(b_tp_init = matrix(0, nparam, 1), b_fp_init = matrix(0, nparam, 1), 
    //                  g_init = g_init, sig2_tp_init = 1/0.35,
    //                  sig2_fp_init = 1/0.2, rho_init = 0,
    //                  e_tp = rnorm(N), e_fp = rnorm(N), e12 = matrix(rnorm(2*N), N, 2), xi = data$xi)
    res[0] = jsem_cpp(ytp, yfp, x_covariates, b_tp, b_fp, g, g_free_ind, sig2_tp, sig2_fp, rho,
                       arma::randn(N), arma::randn(N), arma::mat(N, 2, arma::fill::randn), 
                       xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len);
  }
  else{
    res[omp_get_thread_num()] = jsem_core(ytp, yfp, x_covariates, b_tp, b_fp, g, g_free_ind, sig2_tp, sig2_fp, rho,
                           arma::randn(N), arma::randn(N), arma::mat(N, 2, arma::fill::randn), 
                           xi, beta_tp, alpha_tp, beta_fp, alpha_fp, mcmc_len);
  }
}
  setjsem_threads(old_thread_num);
  return res;
}
