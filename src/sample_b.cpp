// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "arms_ori.h"
#include "jsem_omp.h"


struct log_bj_tp_param{
  arma::vec b_tp;
  unsigned int jj;
  arma::mat ytp;
  arma::uvec xi_3_loc;
  arma::uvec xi_4_loc;
  arma::mat x_covariates;
  arma::mat beta_tp;
  arma::mat alpha_tp;
  arma::vec e_tp;
  arma::vec e1;
};
struct log_bj_fp_param{
  arma::vec b_fp;
  unsigned int jj;
  arma::mat yfp;
  arma::uvec xi_2_loc;
  arma::uvec xi_4_loc;
  arma::mat x_covariates;
  arma::mat beta_fp;
  arma::mat alpha_fp;
  arma::vec e_fp;
  arma::vec e2;
};

// self-defined log_sum_exp to prevent possible underflow / overflow
arma::mat log_sum_exp1_mat(arma::mat tmp){
  arma::mat tmp_zeros = arma::zeros(arma::size(tmp));
  arma::mat max_tmp_0 = arma::max(tmp_zeros, tmp);
  return max_tmp_0 + arma::log(arma::exp(- max_tmp_0) + arma::exp(tmp - max_tmp_0));
}

double log_bj_tp(double x, void* params){
  struct log_bj_tp_param *d;
  d = static_cast<struct log_bj_tp_param *> (params);
  arma::uvec xi_3_loc = d->xi_3_loc;
  arma::uvec xi_4_loc = d->xi_4_loc;
  arma::vec b_tp = d->b_tp;
  b_tp(d->jj) = x;
  
  double res = - x * x / 200;
  
  arma::mat tmp = (d->alpha_tp).rows(xi_3_loc);
  tmp.each_col() %= (d->x_covariates).rows(xi_3_loc) * b_tp +(d->e_tp)(xi_3_loc);
  tmp += (d->beta_tp).rows(xi_3_loc);
  //arma::mat tmp1 = tmp % (d->ytp).rows(xi_3_loc) - log_sum_exp1_mat(tmp);
  arma::mat tmp1 = tmp % (d->ytp).rows(xi_3_loc) - arma::log(1+arma::exp(tmp));
  //res += arma::accu( tmp1.elem(arma::find_finite(tmp1)) );
  res += arma::accu(tmp1);
  
  arma::mat tmp2 = (d->alpha_tp).rows(xi_4_loc);
  tmp2.each_col() %= (d->x_covariates).rows(xi_4_loc) * b_tp + (d->e1)(xi_4_loc);
  tmp2 += (d->beta_tp).rows(xi_4_loc);
  //arma::mat tmp3 = tmp2 % (d->ytp).rows(xi_4_loc) - log_sum_exp1_mat(tmp2);
  arma::mat tmp3 = tmp2 % (d->ytp).rows(xi_4_loc) - arma::log(1+arma::exp(tmp2));
  //res += arma::accu( tmp3.elem(arma::find_finite(tmp3)) );
  res += arma::accu( tmp3 );
  return res;
}
double log_bj_fp(double x, void* params){
  struct log_bj_fp_param *d;
  d = static_cast<struct log_bj_fp_param *> (params);
  arma::uvec xi_2_loc = d->xi_2_loc;
  arma::uvec xi_4_loc = d->xi_4_loc;
  arma::vec b_fp = d->b_fp;
  b_fp(d->jj) = x;
  
  double res = - x * x / 200;
  
  arma::mat tmp = (d->alpha_fp).rows(xi_2_loc);
  tmp.each_col() %= (d->x_covariates).rows(xi_2_loc) * b_fp +(d->e_fp)(xi_2_loc);
  tmp += (d->beta_fp).rows(xi_2_loc);
  //arma::mat tmp1 = tmp % (d->yfp).rows(xi_2_loc) - log_sum_exp1_mat(tmp);
  arma::mat tmp1 = tmp % (d->yfp).rows(xi_2_loc) - arma::log(1+arma::exp(tmp));
  //res += arma::accu( tmp1.elem(arma::find_finite(tmp1)) );
  res += arma::accu( tmp1 );
  
  arma::mat tmp2 = (d->alpha_fp).rows(xi_4_loc);
  tmp2.each_col() %= (d->x_covariates).rows(xi_4_loc) * b_fp + (d->e2)(xi_4_loc);
  tmp2 += (d->beta_fp).rows(xi_4_loc);
  //arma::mat tmp3 = tmp2 % (d->yfp).rows(xi_4_loc) - log_sum_exp1_mat(tmp2);
  arma::mat tmp3 = tmp2 % (d->yfp).rows(xi_4_loc) - arma::log(1+arma::exp(tmp2));
  //res += arma::accu( tmp3.elem(arma::find_finite(tmp3)) );
  res += arma::accu( tmp3 );
  return res;
}
// [[Rcpp::export]]
arma::vec sample_b_tp_c(arma::vec b_tp, const arma::mat &ytp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_tp, 
                        const arma::mat &alpha_tp, const arma::vec &e_tp, const arma::vec &e1){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-6.0, -2.0, 2.0, 6.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_bj_tp_param bj_tp_data;
  bj_tp_data.b_tp = b_tp;
  bj_tp_data.ytp = ytp;
  bj_tp_data.xi_3_loc = arma::find(xi==3);
  bj_tp_data.xi_4_loc = arma::find(xi==4);
  bj_tp_data.x_covariates = x_covariates;
  bj_tp_data.beta_tp = beta_tp;
  bj_tp_data.alpha_tp = alpha_tp;
  bj_tp_data.e_tp = e_tp;
  bj_tp_data.e1 = e1;
  for(unsigned int jj = 0; jj < b_tp.n_elem; ++jj){
    bj_tp_data.jj = jj;
    err = arms(xinit,ninit,&xl,&xr,log_bj_tp,&bj_tp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    //Rprintf("jj=%d, log_b_tp_jj=%f",jj, log_bj_tp())
    bj_tp_data.b_tp(jj) = xsamp[0];
  }
  return bj_tp_data.b_tp;
}
// [[Rcpp::export]]
arma::vec sample_b_fp_c(arma::vec b_fp, const arma::mat &yfp, const arma::uvec &xi, 
                        const arma::mat &x_covariates, const arma::mat &beta_fp, 
                        const arma::mat &alpha_fp, const arma::vec &e_fp, const arma::vec &e2){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xl = -100.0, xr = 100.0;
  double xinit[10]={-6.0, -2.0, 2.0, 6.0}, xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = 0.0;
  
  log_bj_fp_param bj_fp_data;
  bj_fp_data.b_fp = b_fp;
  bj_fp_data.yfp = yfp;
  bj_fp_data.xi_2_loc = arma::find(xi==2);
  bj_fp_data.xi_4_loc = arma::find(xi==4);
  bj_fp_data.x_covariates = x_covariates;
  bj_fp_data.beta_fp = beta_fp;
  bj_fp_data.alpha_fp = alpha_fp;
  bj_fp_data.e_fp = e_fp;
  bj_fp_data.e2 = e2;
  
  for(unsigned int jj = 0; jj < b_fp.n_elem; ++jj){
    bj_fp_data.jj = jj;
    err = arms(xinit,ninit,&xl,&xr,log_bj_fp,&bj_fp_data,&convex,
               npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
    if(err>0){
      Rprintf("error code: %d", err);
      Rcpp::stop("\n");
    }
    bj_fp_data.b_fp(jj) = xsamp[0];
  }
  return bj_fp_data.b_fp;
}
// [[Rcpp::export]]
arma::vec sample_b_linear_reg(arma::vec e_tp1, arma::vec e_tp2, arma::vec e_fp2,
                              arma::mat x_covariates1, arma::mat x_covariates2,
                              arma::vec b_tp, arma::vec b_fp, 
                              double sig2_tp, double sig2_fp, double rho){
  double sig2_0 = 100;
  arma::vec eta_tp1 = x_covariates1 * b_tp + e_tp1;
  arma::vec eta_tp2 = x_covariates2 * b_tp + e_tp2;
  arma::vec eta_fp2 = x_covariates2 * b_fp + e_fp2;
  arma::mat S_n = x_covariates1.t() * x_covariates1 / sig2_tp +
    x_covariates2.t() * x_covariates2 / sig2_tp / (1 - rho * rho);
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_covariates1.t() * eta_tp1 / sig2_tp +
    x_covariates2.t() * eta_tp2 / sig2_tp / (1-rho*rho) - 
    rho/(1-rho*rho)/std::sqrt(sig2_tp * sig2_fp) * x_covariates2.t() * (eta_fp2 - x_covariates2 * b_fp));
  return arma::mvnrnd(mu_n, S_n);
}

arma::vec sample_b_linear_reg_new(arma::vec e1, arma::vec e2,
                                  const arma::mat &x_covariates,
                                  arma::vec b_tp, arma::vec b_fp, 
                                  double sig2_tp, double sig2_fp, double rho){
  double sig2_0 = 100;
  arma::vec eta_tp = x_covariates * b_tp + e1;
  arma::vec eta_fp = x_covariates * b_fp + e2;
  arma::mat S_n = x_covariates.t() * x_covariates / sig2_tp / (1 - rho * rho);
  S_n.diag() += 1.0 / sig2_0;
  S_n = arma::inv_sympd(S_n);
  arma::vec mu_n = S_n * (x_covariates.t() * eta_tp / sig2_tp / (1-rho*rho) - 
    rho/(1-rho*rho)/std::sqrt(sig2_tp * sig2_fp) * x_covariates.t() * (eta_fp - x_covariates * b_fp));
  return arma::mvnrnd(mu_n, S_n);
}



