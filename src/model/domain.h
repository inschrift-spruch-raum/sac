#pragma once

#include "./model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static class LogDomain {
  public:
    static constexpr int scale=256;
    static constexpr int dbits=12;
    static constexpr int dmin=-2047;
    static constexpr int dmax=2047;
    static constexpr int dscale=dmax-dmin+1;
    int fmin,fmax;
    LogDomain()
    {
      for (std::int32_t i=0;i<PSCALE;i++)
      {
        double f=std::max(i,1)/(double)PSCALE;
        double q=std::log(f / (1.0-f))*scale;
        FwdTbl[i]=static_cast<std::int32_t>(std::round(q));
      };
      fmin=FwdTbl[0];
      fmax=FwdTbl[PSCALE-1];
      // 12-Bit
      for (std::int32_t i=dmin;i<=dmax;i++)
      {
        double q=PSCALE/(1.0+std::exp(-double(i)/double(scale)));
        InvTbl[i-dmin]=(int)std::round(q);
      };
    }
    inline std::int32_t Fwd(std::int32_t p)
    {
       return FwdTbl[p];
    }
    inline std::int32_t Inv(std::int32_t x)
    {
       if (x<dmin) return 1;
       else if (x>dmax) return PSCALEm;
       else return InvTbl[x-dmin];
    }
    void Check()
    {
      std::int32_t sum=0;
      for (std::int32_t i=0;i<PSCALE;i++)
      {
        std::int32_t p=Inv(Fwd(i));
        sum+=(p-i)*(p-i);
      }
      std::printf(" mse: %0.4f\n",double(sum)/double(PSCALE));
    }
  protected:
    std::int32_t FwdTbl[PSCALE];
    std::int32_t InvTbl[dscale];
} myDomain;
