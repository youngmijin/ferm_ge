#include <cfloat>
#include <cmath>
#include <random>

using std::minstd_rand;
using std::random_device;
using std::uniform_real_distribution;

typedef struct {
  size_t T;                // Number of iterations
  size_t thr_granularity;  // Number of threshold candidates

  double D_bar;       // Time-averaged hypothesis values
  double lambda_bar;  // Time-averaged lambda values

  size_t *hypi_stat;  // Number of times each hypothesis index was selected
  size_t *hypi_t;     // (opt) History of hypothesis index values
} C_GEFAIR_RESULT;

inline size_t find_thri(size_t thr_candidates_len, double *thr_candidates,
                        double *I_alpha_cache, double *err_cache, double lambda,
                        double gamma) {
  /* Finds threshold for a given lambda value (the "oracle") */
  double thri_of_lambda = 0.0;
  double min_L_value = DBL_MAX;
  for (size_t i = 0; i < thr_candidates_len; i += 1) {
    double L_value = err_cache[i] + lambda * (I_alpha_cache[i] - gamma);
    if (L_value < min_L_value) {
      min_L_value = L_value;
      thri_of_lambda = i;
    }
  }
  return thri_of_lambda;
}

inline double get_Iup(double alpha, double c, double a) {
  /* Calculates Iup (maximum available I_alpha value) */
  double ca = (c + a) / (c - a);
  if (alpha == 0)
    return log(ca);
  else if (alpha == 1)
    return ca * log(ca);
  else
    return (pow(ca, alpha) - 1) / abs(alpha * (alpha - 1));
}

extern "C" {
C_GEFAIR_RESULT *solve_gefair(size_t thr_candidates_len, double *thr_candidates,
                              double *I_alpha_cache, double *err_cache,
                              double lambda_max, double nu, double alpha,
                              double gamma, double c, double a) {
  /*
    Solves GE-Fairness (reference: algorithm 1 in the paper)
    Note that hypothesis is a float value in this implementation.
  */

  double A_alpha = 1 + lambda_max * (gamma + get_Iup(alpha, c, a));
  double B = gamma * lambda_max;

  size_t T = 4 * A_alpha * A_alpha * log(2) / (nu * nu);
  double kappa = nu / (2 * A_alpha);

  double w0 = 1.0;
  double w1 = 1.0;

  double lambda_0 = 0.0;
  double lambda_1 = lambda_max;

  // implementation hack: avoid repeated calculation of multiplicative factors
  double *w0_mult = (double *)malloc(thr_candidates_len * sizeof(double));
  double *w1_mult = (double *)malloc(thr_candidates_len * sizeof(double));
  for (size_t i = 0; i < thr_candidates_len; i += 1) {
    w0_mult[i] = pow(kappa + 1.0, (err_cache[i] + B) / A_alpha);
    w1_mult[i] =
        pow(kappa + 1.0,
            ((err_cache[i] + lambda_max * (I_alpha_cache[i] - gamma)) + B) /
                A_alpha);
  }

  // implementation hack: pre-calculate oracle because the lambda will be chosen
  //                      from a discrete set of values
  size_t lambda_0_thri = find_thri(thr_candidates_len, thr_candidates,
                                   I_alpha_cache, err_cache, 0.0, gamma);
  size_t lambda_1_thri = find_thri(thr_candidates_len, thr_candidates,
                                   I_alpha_cache, err_cache, lambda_max, gamma);

  minstd_rand rng(random_device{}());
  uniform_real_distribution<double> dist(0.0, 1.0);

  double hyp_sum = 0.0;
  double lambda_sum = 0.0;

  size_t *hypi_stat = (size_t *)calloc(thr_candidates_len, sizeof(size_t));

#ifdef TRACE_HYPI_T
  size_t *hypi_t = (size_t *)malloc(T * sizeof(size_t));
#endif

  // implementation hack: use the "index number" of the threshold instead of
  //                      the threshold itself throughout the algorithm
  for (size_t t = 0; t < T; t += 1) {
    // 1. destiny chooses lambda_t
    int is_0_chosen = int(dist(rng) < (w0 / (w0 + w1)));
    double lambda_t = is_0_chosen * lambda_0 + (1 - is_0_chosen) * lambda_1;
    lambda_sum += lambda_t;

    // 2. the learner chooses a hypothesis (threshold(float) in this case)
    size_t thri_t =
        is_0_chosen * lambda_0_thri + (1 - is_0_chosen) * lambda_1_thri;
    hyp_sum += thr_candidates[thri_t];
    hypi_stat[thri_t] += 1;
#ifdef TRACE_HYPI_T
    hypi_t[t] = thri_t;
#endif

    // 3. destiny updates the weight vector (w0, w1)
    w0 = w0 * w0_mult[thri_t];
    w1 = w1 * w1_mult[thri_t];
  }

  auto result = (C_GEFAIR_RESULT *)malloc(sizeof(C_GEFAIR_RESULT));
  result->T = T;
  result->thr_granularity = thr_candidates_len;
  result->D_bar = hyp_sum / T;
  result->lambda_bar = lambda_sum / T;
  result->hypi_stat = hypi_stat;

#ifdef TRACE_HYPI_T
  result->hypi_t = hypi_t;
#else
  result->hypi_t = nullptr;
#endif

  free(w0_mult);
  free(w1_mult);

  return result;
}

void free_gefair_result(C_GEFAIR_RESULT *result) {
#ifdef TRACE_HYPI_T
  free(result->hypi_t);
#endif
  free(result->hypi_stat);
  free(result);
}

#ifdef TRACE_HYPI_T
bool FLAG_TRACE_HYPI_T = true;
#else
bool FLAG_TRACE_HYPI_T = false;
#endif

size_t SIZE_OF_DOUBLE = sizeof(double);
size_t SIZE_OF_SIZE_T = sizeof(size_t);
}
