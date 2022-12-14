#ifndef __ARMS_ORI__
#define __ARMS_ORI__
#include <RcppArmadillo.h>

int arms (double *xinit, int ninit, double *xl, double *xr, 
          double (*myfunc)(double x, void *mydata), void *mydata,
          double *convex, int npoint, int dometrop, double *xprev, double *xsamp,
          int nsamp, double *qcent, double *xcent,
          int ncent, int *neval);

#endif