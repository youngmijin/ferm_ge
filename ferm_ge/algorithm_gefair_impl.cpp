#include <cfloat>
#include <cmath>
#include <map>
#include <random>

using std::map;
using std::minstd_rand;
using std::pair;
using std::random_device;
using std::uniform_real_distribution;

typedef struct {
  size_t T;
  size_t *D_bar;
  double *lambda_bar;
  double *hypothesis_history;
  double *I_alpha_history;
  double *err_history;
} GEFAIR_RESULT;

/* GE-Fairness Solver */

inline double calc_L(map<double, pair<double, double>> *ge_err_cache,
                     double threshold, double lambda, double gamma) {
  // Find Lagrangian with given lambda and threshold
  auto cached_value = ge_err_cache->find(threshold);
  double I_alpha = cached_value->second.first;
  double err = cached_value->second.second;
  return err + lambda * (I_alpha - gamma);
}

inline double find_threshold(map<double, pair<double, double>> *ge_err_cache,
                             map<double, double> *thr_cache,
                             size_t thr_candidates_size, double *thr_candidates,
                             double lambda, double gamma) {
  // Find threshold for a given lambda value (the "oracle")
  auto cached_value = thr_cache->find(lambda);
  if (cached_value != thr_cache->end()) return cached_value->second;

  double thr_of_lambda = 0.0;
  double min_L_value = DBL_MAX;
  for (size_t i = 0; i < thr_candidates_size; i += 1) {
    double L_value = calc_L(ge_err_cache, thr_candidates[i], lambda, gamma);
    if (L_value < min_L_value) {
      min_L_value = L_value;
      thr_of_lambda = thr_candidates[i];
    }
  }

  thr_cache->insert({lambda, thr_of_lambda});
  return thr_of_lambda;
}

inline double get_Iup(double alpha, double r) {
  // Calculate Iup (reference: section 5 in the paper)
  double rr = (r + 1) / (r - 1);
  if (alpha == 0)
    return log(rr);
  else if (alpha == 1)
    return rr * log(rr);
  else
    return (pow(rr, alpha) - 1) / abs(alpha * (alpha - 1));
}

extern "C" GEFAIR_RESULT *solve_gefair(size_t thr_candidates_size,
                                       double *thr_candidates,
                                       double *I_alpha_cache, double *err_cache,
                                       double alpha, double lambda_max,
                                       double nu, double r, double gamma,
                                       bool collect_ge_history) {
  double A_alpha = 1 + lambda_max * (gamma + get_Iup(alpha, r));
  double B = gamma * lambda_max;

  size_t T = 4 * A_alpha * A_alpha * log(2) / (nu * nu);
  double kappa = nu / (2 * A_alpha);

  map<double, double> thr_cache;  // this map will get filled up during the
                                  // algorithm

  map<double, pair<double, double>> ge_err_cache;
  for (size_t i = 0; i < thr_candidates_size; i += 1)
    ge_err_cache.insert({thr_candidates[i], {I_alpha_cache[i], err_cache[i]}});

  map<double, double> w0_mult_cache;
  for (size_t i = 0; i < thr_candidates_size; i += 1)
    w0_mult_cache.insert(
        {thr_candidates[i],
         pow(kappa + 1.0,
             (calc_L(&ge_err_cache, thr_candidates[i], 0, gamma) + B) /
                 A_alpha)});

  map<double, double> w1_mult_cache;
  for (size_t i = 0; i < thr_candidates_size; i += 1)
    w1_mult_cache.insert(
        {thr_candidates[i],
         pow(kappa + 1.0,
             (calc_L(&ge_err_cache, thr_candidates[i], lambda_max, gamma) + B) /
                 A_alpha)});

  minstd_rand rng(random_device{}());
  uniform_real_distribution<double> dist(0.0, 1.0);

  /*
    Solve GE-Fairness (reference: algorithm 1 at section 5 in the paper)
    Note that hypothesis is a float value in this implementation.
    After solving, the function returns the hypothesis choice counts and the
    lambda choice history.
  */

  double w0 = 1.0;
  double w1 = 1.0;

  double lambda_0 = 0.0;
  double lambda_1 = lambda_max;

  double *hypothesis_history = (double *)malloc(T * sizeof(double));
  double *lambda_bar = (double *)malloc(T * sizeof(double));

  for (size_t t = 0; t < T; t += 1) {
    double random_t = w0 / (w0 + w1);
    double lambda_t = (dist(rng) < random_t) ? lambda_0 : lambda_1;
    double thr_t =
        find_threshold(&ge_err_cache, &thr_cache, thr_candidates_size,
                       thr_candidates, lambda_t, gamma);

    double w0_mult = w0_mult_cache.find(thr_t)->second;
    double w1_mult = w1_mult_cache.find(thr_t)->second;

    w0 = w0 * w0_mult;
    w1 = w1 * w1_mult;

    hypothesis_history[t] = thr_t;
    lambda_bar[t] = lambda_t;
  }

  // get statistics of hypothesis choices (so called D_bar)
  size_t *D_bar = (size_t *)malloc(thr_candidates_size * sizeof(size_t));
  for (size_t i = 0; i < thr_candidates_size; i += 1) {
    D_bar[i] = 0;
    for (size_t t = 0; t < T; t += 1) {
      if (hypothesis_history[t] == thr_candidates[i]) D_bar[i] += 1;
    }
  }

  /* GE Fairness Solver End */

  auto result = (GEFAIR_RESULT *)malloc(sizeof(GEFAIR_RESULT));
  result->T = T;
  result->D_bar = D_bar;
  result->lambda_bar = lambda_bar;

  if (collect_ge_history) {
    double *I_alpha_history = (double *)malloc(T * sizeof(double));
    double *err_history = (double *)malloc(T * sizeof(double));
    for (size_t t = 0; t < T; t += 1) {
      double thr_t = hypothesis_history[t];
      auto ge_err = ge_err_cache.find(thr_t);
      I_alpha_history[t] = ge_err->second.first;
      err_history[t] = ge_err->second.second;
    }
    result->I_alpha_history = I_alpha_history;
    result->err_history = err_history;
    result->hypothesis_history = hypothesis_history;
  } else {
    free(hypothesis_history);
    result->I_alpha_history = nullptr;
    result->err_history = nullptr;
    result->hypothesis_history = nullptr;
  }

  return result;
}

extern "C" void free_gefair_result(GEFAIR_RESULT *result) {
  free(result->D_bar);
  free(result->lambda_bar);
  if (result->I_alpha_history != nullptr) free(result->I_alpha_history);
  if (result->err_history != nullptr) free(result->err_history);
  if (result->hypothesis_history != nullptr) free(result->hypothesis_history);
  free(result);
}
