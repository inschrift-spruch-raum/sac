#pragma once // SPARSEPCM_H

#include "../common/math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include <numeric>
#include "../common/utils.h"

class SimplePred {
public:
  SimplePred() = default;

  double Predict() const { return lb; }

  void Update(std::int32_t val) { lb = val; }

protected:
  std::int32_t lb{0};
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

    struct Stats {
      double fraction_used=0,fraction_cost=0;
    } st;
    int32_t minval,maxval;
    std::vector<int>used;
  private:
    int GetBaseRank(const int32_t p) const;
    std::vector<int>prefix,inv_prefix;
};
