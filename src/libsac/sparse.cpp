#include "sparse.h"

SparsePCM::SparsePCM()
:minval(0),maxval(0)
{
}

void SparsePCM::BuildPrefixSums()
{
  //create prefix sum, count used values before "i"
  prefix.assign(used.size()+1,0);
  for (size_t i=0;i<used.size();++i) {
    prefix[i+1]=prefix[i]+used[i];
  }

  inv_prefix.assign(prefix.back(),0);
  for (std::size_t i=0;i<used.size();++i) {
    if (used[i])
      inv_prefix[prefix[i]]=i;
  }
}

void SparsePCM::SetRanges(int32_t minimum_val,int32_t maximum_val)
{
  minval = minimum_val;
  maxval = maximum_val;
  used.assign(maxval-minval+1,0);
}

void SparsePCM::Analyse(span_ci32 buf)
{
  st.fraction_used = st.fraction_cost = 0.0;
  auto [min_it, max_it] = std::minmax_element(std::begin(buf),std::end(buf));
  SetRanges(*min_it,*max_it);

  for (auto val : buf) used[val-minval] = 1;
  int sum=std::accumulate(begin(used),end(used),0);
  st.fraction_used = used.size()>0?(sum/static_cast<double>(used.size()))*100.:0.0;

  BuildPrefixSums();
  //std::cout << "total sum: " << sum << " prefix " << prefix.back() << '\n';

  // calc cost
  //SimplePred pred;
  double sum0=0,sum1=0;
  for (auto val : buf) {
    //int32_t p=std::clamp((int)std::round(pred.Predict()),minval,maxval);
    int32_t e0=val;
    //int32_t e1=val2rank(e0);
    int32_t e1=Map(e0);
    //if (e1!=e2)
    //  std::cerr << "  warning: " << e1 << ' ' << e2 << '\n';

    sum0+=pow(std::fabs(e0),cost_pow);
    sum1+=pow(std::fabs(e1),cost_pow);
    //pred.Update(val);
  }
  st.fraction_cost=sum1>0?sum0/static_cast<double>(sum1):0;
}

int SparsePCM::GetBaseRank(const int32_t p) const
{
  const int N=static_cast<int>(used.size());
  const int pidx = p - minval;// pivot/prediction index

  //base = rank(first used symbol >= p);
  const int pidx_clamp=std::clamp(pidx,0,N);
  const int ceil_rank = prefix[pidx_clamp];

  return ceil_rank;
}

// map error "val" (relative to prediction p) to the rank distance among only used symbols
// mapping is bijective, val+p must be in [minval,maxval], p can be outside
int SparsePCM::Map(const int32_t val,const int32_t p) const
{
  const int tidx = (val+p) - minval; // target index
  if (tidx < 0 || tidx>=static_cast<int>(used.size()) || used[tidx]==0)
    throw std::runtime_error("SparsePCM Map: target index outside support");

  return prefix[tidx] - GetBaseRank(p);
}

int32_t SparsePCM::Unmap(const int32_t mrank,const int32_t p) const
{
  const int idx = GetBaseRank(p) + mrank;
  if (idx < 0 || idx >= static_cast<int>(inv_prefix.size()))
    throw std::runtime_error("SparsePCM Unmap: rank outside support");
  return inv_prefix[idx] + minval - p;
}
