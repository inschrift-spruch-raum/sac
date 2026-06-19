#pragma once // LS_H

#include "../global.h"
#include "../common/math.h"
#include <algorithm>
#include <cmath>

// Linear systems using gradient descent (ADAGrad/RMsprop style)
// supports: custom loss function, weight decay and init type
enum class LSInitType {Zero,One,Uniform,Decay};

//LossFunction
namespace Loss {

template<typename T>
concept LossFunction = requires(double err) {
  { T::grad(err) } -> std::same_as<double>;
};

struct L1 {
  static inline double grad(double err){return MathUtils::sgn(err);}};
struct L2 {static inline double grad(double err) {return err;}};
template <double d=4>
  struct HBR {static inline double grad(double err) {
    return MathUtils::hbr_grad(err,d);}
  };

};

//Regularization
namespace Reg {

template<typename T>
concept WeightDecay = requires(double w,double mu) {
  { T::apply(w,mu) } -> std::same_as<double>;
};

struct None {
  static inline double apply(double w,double) {
      return w;
    }
};

template <double nu=0.001>
struct L1 {
  static inline double apply(double w,double mu)
  {
    double t=mu*nu;
    //soft-thresholding
    if (w> t) return w-t;
    if (w<-t) return w+t;
    return 0;
  };
};

template <double nu=0.001>
struct L2 {
  static inline double apply(double w,double mu) {
    return w*(1.0-mu*nu);
  }
};
};

class LS {
  protected:
  public:
    LS(std::size_t n,double mu)
    :n(n),w(n),mu(mu)
    {
    }
    virtual double Predict(span_cf64 x)
    {
      assert(x.size() == static_cast<size_t>(n));
      return MathUtils::dot_scalar(x,w);
    }
    virtual void Update(span_cf64,double)=0;
    virtual double GetWeight(std::int32_t idx) const
    {
      return w[idx];
    }
    virtual ~LS() = default ;
  protected:
    static void InitWeights(vec1D &weights,LSInitType init_type)
    {
      switch (init_type) {
        case LSInitType::Zero:
          std::ranges::fill(weights,0.0);
          break;
        case LSInitType::One:
          std::ranges::fill(weights,1.0);
          break;
        case LSInitType::Uniform:
          std::ranges::fill(weights,1.0/weights.size());
          break;
        case LSInitType::Decay:
          for (std::size_t i=0;i<weights.size();++i)
            weights[i]=1.0/(1.0+i);
          break;
      }
    }
    std::size_t n;
    vec1D w;
    double mu;
};


template<Loss::LossFunction LF=Loss::L2,LSInitType init_type=LSInitType::Zero,Reg::WeightDecay WD=Reg::None>
class LS_ADA : public LS
{
  public:
    LS_ADA(std::int32_t n,double mu,double beta=0.95)
    :LS(n,mu),eg(n),beta(beta),beta1(1.0-beta)
    {
      InitWeights(w,init_type);
    }
    void Update(span_cf64 x,double error) override {
      const double loss=LF::grad(error);

      for (std::size_t i=0;i<n;++i) {
        double const grad=loss*x[i];

        eg[i]=beta*eg[i]+beta1*grad*grad; //ema gradients

        double mu_scaled = mu/(sqrt(eg[i])+SACCfg::LMS_ADA_EPS);
        w[i]+=mu_scaled*grad;

        //weight decay (proximal step for L1)
        w[i]=WD::apply(w[i],mu_scaled);
      }
    }
  private:
    vec1D eg;
    double beta,beta1;
};

template<Loss::LossFunction LF=Loss::L2,LSInitType init_type=LSInitType::Zero,Reg::WeightDecay WD=Reg::None>
class LS_ADAM : public LS
{
  public:
    LS_ADAM(std::int32_t n,double mu,double beta1=0.9,double beta2=0.95)
    :LS(n,mu),M(n),S(n),beta1(beta1),beta2(beta2)
    {
      InitWeights(w,init_type);
      power_beta1=1.0;
      power_beta2=1.0;
    }
    void Update(span_cf64 x,double error) override {
      power_beta1*=beta1;
      power_beta2*=beta2;
      const double loss=LF::grad(error);
      for (std::size_t i=0;i<n;++i) {
        double const grad=loss*x[i]; // gradient

        M[i]=beta1*M[i]+(1.0-beta1)*grad;
        S[i]=beta2*S[i]+(1.0-beta2)*(grad*grad);

        //bias correction
        double m_hat=M[i]/(1.0-power_beta1);
        double n_hat=S[i]/(1.0-power_beta2);
        w[i]+=mu*m_hat/(sqrt(n_hat)+SACCfg::LMS_ADA_EPS);

        //weight decay
        w[i]=WD::apply(w[i],mu);
      }
    }
  private:
    vec1D M,S;
    double beta1,beta2,power_beta1,power_beta2;
};

// sign-sign lms algorithm
class SSLMS : public LS {
  public:
      SSLMS(std::int32_t n,double mu)
      :LS(n,mu)
      {
      }
      void Update(span_cf64 x,double error) override
      {
        const double wf=mu*MathUtils::sgn(error);
        for (std::size_t i=0;i<n;++i) {
           w[i]+=wf*MathUtils::sgn(x[i]);
        }
      }
};
