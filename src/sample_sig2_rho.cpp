// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
#include <RcppDist.h>
#include "arms_ori.h"

struct log_sig2_param{
  double sig2_fp_tp;
  double rho;
  arma::vec e_tp_fp;
  arma::mat e12;
};
struct log_rho_params{
  arma::mat e12;
  double sig2_tp;
  double sig2_fp;
};

double log_sig2_tp_c(double x, void* params){
  struct log_sig2_param *d;
  d = static_cast<struct log_sig2_param *> (params);
  double sig2_fp = (d->sig2_fp_tp);
  arma::vec e_tp = (d->e_tp_fp);
  double rho = d->rho;
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;

  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);

  double res = -(inv_gamma_alpha + 1 + 0.5 * 2 * e_tp.n_elem) * std::log(x);
  res += - ( 0.5 * arma::accu(arma::square(e_tp)) + inv_gamma_beta) / x;
  res += - 0.5 / (1-rho * rho) * (arma::accu(arma::square(e1)) / x -
    2*rho*arma::accu(e1 % e2) / std::sqrt(x * sig2_fp));
  return res;
}
double log_sig2_fp_c(double x, void* params){
  struct log_sig2_param *d;
  d = static_cast<struct log_sig2_param *> (params);
  double sig2_tp = (d->sig2_fp_tp);
  arma::vec e_fp = (d->e_tp_fp);
  double rho = d->rho;
  double inv_gamma_alpha = 0.01;
  double inv_gamma_beta = 0.01;
  
  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);
  
  double res = -(inv_gamma_alpha + 1 + 0.5 * 2 * e_fp.n_elem) * std::log(x);
  res += - ( 0.5 * arma::accu(arma::square(e_fp)) + inv_gamma_beta) / x;
  res += - 0.5 / (1-rho * rho) * (arma::accu(arma::square(e2)) / x -
    2*rho*arma::accu(e1 % e2) / std::sqrt(x * sig2_tp));
  return res;
}
// [[Rcpp::export]]
double sample_sig2_tp_c(double sig2_tp, double sig2_fp, double rho,
                      arma::vec e_tp, arma::mat e12, arma::uvec xi){
  
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_tp;
  
  log_sig2_param log_sig2_data;
  log_sig2_data.sig2_fp_tp = sig2_fp;
  log_sig2_data.rho = rho;
  log_sig2_data.e_tp_fp = e_tp;
  log_sig2_data.e12 = e12;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_tp_c,&log_sig2_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}
// [[Rcpp::export]]
double sample_sig2_fp_c(double sig2_tp, double sig2_fp, double rho,
                        arma::vec e_fp, arma::mat e12, arma::uvec xi){
  
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10]={0.5, 2.0, 4.0, 8.0}, xl = 0.0, xr = 100.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1;
  double xprev = sig2_fp;
  
  log_sig2_param log_sig2_data;
  log_sig2_data.sig2_fp_tp = sig2_tp;
  log_sig2_data.rho = rho;
  log_sig2_data.e_tp_fp = e_fp;
  log_sig2_data.e12 = e12;
  
  err = arms(xinit,ninit,&xl,&xr,log_sig2_fp_c,&log_sig2_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}
 
double log_rho_c(double x, void* params){
  struct log_rho_params *d;
  d = static_cast<struct log_rho_params *> (params);
  arma::vec e1 = (d->e12).col(0);
  arma::vec e2 = (d->e12).col(1);
  double sig2_tp = d->sig2_tp;
  double sig2_fp = d->sig2_fp;
  double res = -0.5 * e1.n_elem * std::log( 1 - x * x) -
    0.5 / (1 - x * x) * 
    (arma::accu(arma::square(e1)) / sig2_tp + arma::accu(arma::square(e2)) / sig2_fp -
    2* x * arma::accu(e1 % e2) / std::sqrt(sig2_tp * sig2_fp));
  
  return res;
}
// [[Rcpp::export]]
double sample_rho_c(double rho, const arma::uvec &xi, double sig2_tp, double sig2_fp, const arma::mat &e12){
  int err, ninit = 4, npoint = 100, nsamp = 1, ncent = 0 ;
  int neval;
  double xinit[10] = {-0.4, 0.2, 0.4, 0.6}, xl = -1.0, xr = 1.0;
  double xsamp[100], xcent[10], qcent[10] = {5., 30., 70., 95.};
  double convex = 1.;
  int dometrop = 1; // this is a log-concave function
  double xprev = rho;
  
  log_rho_params log_rho_data;
  log_rho_data.sig2_tp = sig2_tp;
  log_rho_data.sig2_fp = sig2_fp;
  log_rho_data.e12 = e12;
  
  err = arms(xinit, ninit,&xl,&xr,log_rho_c,&log_rho_data,&convex,
             npoint,dometrop,&xprev,xsamp,nsamp,qcent,xcent,ncent,&neval);
  if(err>0){
    Rprintf("error code: %d", err);
    Rcpp::stop("\n");
  }
  return xsamp[0];
}
//' @export
// [[Rcpp::export]]
arma::mat sample_Sigma_new(const arma::mat &e12){
  arma::mat S = arma::eye(2,2) + e12.t() * e12;
  return riwish(2 + e12.n_rows, S);
}


