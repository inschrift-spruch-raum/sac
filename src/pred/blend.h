#pragma once

#include "../global.h"
#include "../common/math.h"
#include "../common/utils.h"

// blend two expert outputs via sigmoid
class Blend2 {
  public:
    Blend2(double beta=0.95,double theta0=1.0,double theta1=0.0,double scale=5.0)
    :w(0.5),
     th0(theta0),th1(theta1),scale(scale),
     rsum(beta)
    {
    }
    double Predict(double p0,double p1) const
    {
      return w*p0 + (1.0-w)*p1;
    }

    void Update(double score0,double score1)
    {
      rsum.Update(score1-score0);

      double delta=rsum.Get();
      double z = th0*delta + th1;
      z = std::clamp(z,-scale,scale);
      w = 1.0 / (1.0 + std::exp(-z));
   }
  private:
    double w,th0,th1,scale;
    RunSumEMA rsum;
};

constexpr bool BLEND_MV = true;

namespace detail {
  inline double computeZScore(const RunMeanVar& r, double beta, double EPS) {
    auto [mean, var] = r.Get();
    return beta * mean / (std::sqrt(var) + EPS);
  }

  inline double
  computeZScore(const RunSumEMA& r, double beta, double /*unused*/) {
    return beta * r.Get(); // EPS 在此无意义，省略参数名
  }

  static auto createRuner(double alpha) {
    if constexpr(BLEND_MV) {
      return RunMeanVar(alpha);
    } else {
      return RunSumEMA(alpha);
    }
  }
} // namespace detail

class BlendExp
{
  static constexpr double EPS=1E-8;
  public:
    BlendExp(std::size_t n,double alpha,double beta)
    :n_(n),beta(beta),px(0.0),
    x(n),w(n),zm(n),
    rsum(n,detail::createRuner(alpha))
    {
      if(n != 0U) {
        std::ranges::fill(w, 1.0 / static_cast<double>(n)); // init equal weight
      }
    };
    double Predict(const vec1D &input)
    {
      x=input;
      px=MathUtils::dot_scalar(x,w);
      return px;
    }
    void Update(double target)
    {
      UpdateScores(target);
      UpdateWeights();
    }
    const vec1D &Weights()const {return w;}
  private:
    void UpdateScores(double target)
    {
      double loss_px = std::abs(target-px);
      for (std::size_t i=0;i<n_;i++) {
        double loss_pi=std::abs(target-x[i]);
        // if score > 0 -> expert better then blend
        double score=(loss_px - loss_pi);
        rsum[i].Update(score);
      }
    }
    // softmax w_i = exp(-beta * normalized_regret)
    void UpdateWeights()
    {
      double max_z = -std::numeric_limits<double>::infinity();
      for (std::size_t i=0;i<n_;i++) {
        zm[i] = detail::computeZScore(rsum[i], beta, EPS);
        max_z = std::max(max_z, zm[i]);
      }

      //best expert has highest z-score -> weight=exp(0)=1
      double total=0.0;
      for (std::size_t i=0;i<n_;i++) {
        w[i] = std::exp(zm[i]-max_z);
        total += w[i];
      }
      //normalize weights, total >= 1 from max-trick
      const double inv_total=1.0/total;
      for (double &val : w) {
        val *= inv_total;
      }
    }

    std::size_t n_;
    double beta,px;
    vec1D x,w,zm;
    using RunerType = decltype(detail::createRuner(std::declval<double>()));
    std::vector <RunerType> rsum;
};
