#pragma once // LPC_H

#include "../common/utils.h"
#include "../common/math.h"
#include <type_traits>

class OLS {
  public:
    explicit OLS(std::int32_t n,std::int32_t kmax=1,double lambda=0.998,double nu=0.001,double beta_sum=0.6,double beta_pow=0.75,double beta_add=2);
    double Predict();
    void Update(double val);
    vec1D x;
  private:
    using Decomp = std::conditional_t<
      SACCfg::USE_LDLT,
      MathUtils::LDLT,
      MathUtils::Cholesky
    >;
    Decomp decomp;
    vec1D w,b;
    vec2D mcov;
    std::int32_t n,kmax,km;
    double lambda,nu,pred;
    double beta_pow,beta_add,w_decay;
    RunSumGEO esum;
};
