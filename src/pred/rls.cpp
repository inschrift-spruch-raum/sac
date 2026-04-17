#include "rls.h"
#include "../common/math.h"
#include "../common/utils.h"

constexpr std::int32_t RLS_ALC=1;

RLS::RLS(std::int32_t n,double gamma,double beta,double nu)
:n(n),
px(0.),
x(n),w(n),
P(n,vec1D(n)), // inverse covariance matrix
alc(gamma,beta)
{
  for (std::int32_t i=0;i<n;i++) {
    P[i][i]=1.0/nu;
  }
}

double RLS::Predict()
{
  px=MathUtils::dot_scalar(x,w);
  return px;
}

double RLS::Predict(const vec1D &input)
{
  x=input;
  return Predict();
}

void RLS::Update(double val)
{
  const double err=val-px;
  
  // a priori variance of prediction, //phi=x^T P x
  vec1D ph=MathUtils::mul(P,x);
  const double phi=std::max(MathUtils::dot_scalar(x,ph),PHI_FLOOR);

  // adaptive lambda control using
  // Normalized Innovation Squared (NIS)
  // quantifies how "unexpected" the observation is
  // relative to the models uncertainty phi
  double alpha=alc.Get(err*err,phi);

  //update inverse of covariance matrix
  //P(n)=1/lambda*P(n-1)-1/lambda * k(n)*x^T(n)*P(n-1)
  double denom=1./(alpha+phi);
  double inv_alpha=1.0/(alpha);
  for(std::int32_t i = 0; i < n; i++) {
    for(std::int32_t j = 0; j <= i; j++) {
      double m = ph[i] * ph[j]; // outer product of ph
      double v = (P[i][j] - denom * m) * inv_alpha;
      P[i][j] = P[j][i] = v;
    }
  }

  // update weights
  for(std::int32_t i = 0; i < n; i++) {
    w[i] += err * (denom * ph[i]);
  }
  alc.Update(err*err,phi);
}

void RLS::UpdateHist(double val)
{
  Update(val);
  miscUtils::RollBack(x,val);
}
