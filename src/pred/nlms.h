#ifndef NLMS_H
#define NLMS_H

#include "../global.h"
#include "../common/histbuf.h"

class NLMS_Stream
{
  public:
    NLMS_Stream(int n,double mu,double mu_decay=1.0,double pow_decay=0.8)
    :n(n),mu(mu),
    x(n),w(n),mutab(n),powtab(n)
    {
      pred=0.0;
      sum_powtab=0.0;
      for (int i=0;i<n;i++) {
         powtab[i]=1.0/(pow(1+i,pow_decay));
         sum_powtab+=powtab[i];
         mutab[i]=pow(mu_decay,i);
      }
    }
    double Predict()
    {
      pred=slmath::dot(x.get_span(),w);
      return pred;
    }
    void Update(double val)
    {
      const double spow=slmath::calc_s2pow(x.get_span(),powtab);
      const double wgrad=mu*(val-pred)*sum_powtab/(spow+SACCfg::NLMS_POW_EPS);
      for (int i=0;i<n;i++) {
        w[i]+=mutab[i]*(wgrad*x[i]);
        if constexpr(SACCfg::NLMS_CLAMPW)
          w[i]=std::clamp(w[i],-SACCfg::NLMS_SCALE,SACCfg::NLMS_SCALE);

      }
      x.push(val);
    };
  protected:
    int n;
    double mu;
    RollBuffer2<double>x;
    std::vector<double,align_alloc<double>> w,mutab,powtab;
    double sum_powtab,pred;
};

#endif

