#ifndef SPARSEPCM_H
#define SPARSEPCM_H

#include <numeric>
#include "../common/utils.h"

class SimplePred {
  public:
    SimplePred()
    :lb(0)
    {

    }
    double Predict()
    {
      return lb;
    }
    void Update(int32_t val)
    {
      lb = val;
    }
  protected:
    int32_t lb;
};

// rank-mapping errors
// counting how many used symbols exist between the prediction and the target: error+prediction
class SparsePCM {
  const double cost_pow=1;
  public:
    SparsePCM();
    void Analyse(span_ci32 buf);
    void SetRanges(int32_t minimum_val,int32_t maximum_val);
    void BuildPrefixSums();
    int Map(const int32_t val,const int32_t p=0) const;
    int32_t Unmap(const int32_t mrank,const int32_t p=0) const;

    /*
    //possible oob
    int val2rank(const int32_t val,const int32_t p=0)
    {
      if (val==0) return 0;
      const int sgn=MathUtils::sgn(val);

      const int pidx=p-minval;
      int mres=0;
      if (val>0) {
        for (int i=pidx+1;i<=pidx+val;i++)
          mres+=used[i];
          //if (used[i]) ++mres;
      } else {
        for (int i=pidx-1;i>=pidx+val;i--)
          mres+=used[i];
         //if (used[i])  ++mres;
      }
      return sgn*mres;
    }*/
    struct Stats {
      double fraction_used=0,fraction_cost=0;
    } st;
    int32_t minval,maxval;
    std::vector<int>used;
  private:
    int GetBaseRank(const int32_t p) const;
    std::vector<int>prefix,inv_prefix;
};


#endif // SPARSEPCM_H
