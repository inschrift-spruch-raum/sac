#include "lpc.h"

constexpr bool INIT_COV = false;

OLS::OLS(int n,int kmax,double lambda,double nu,double beta_sum,double beta_pow,double beta_add)
:x(n),
chol(n),
w(n),b(n),mcov(n,vec1D(n)),
n(n),kmax(kmax),km(0), lambda(lambda),nu(n*nu),
pred(0.0), beta_pow(beta_pow),beta_add(beta_add),esum(beta_sum)
{
  if constexpr (INIT_COV) {
    for (std::int32_t i=0;i<n;i++) { mcov[i][i]=1.0;}
  }
}

double OLS::Predict()
{
  pred = MathUtils::dot_scalar(span_cf64(x.data(), n), span_cf64(w.data(), n));
  return pred;
}

void OLS::Update(double val)
{
  // update estimate of covariance matrix
  esum.Update(fabs(val-pred));
  double c0=std::pow(esum.Get()+beta_add,-beta_pow);

  for (std::int32_t j=0;j<n;j++) {
    // only update lower triangular
    for (std::int32_t i=0;i<=j;i++) { mcov[j][i]=lambda*mcov[j][i]+c0*(x[j]*x[i]);}
    b[j]=lambda*b[j]+c0*(x[j]*val);
  }

  km++;
  if (km>=kmax) {
    if (chol.Factor(mcov,nu) == 0) { chol.Solve(b,w);}
    km=0;
  }
}
