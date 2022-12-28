#include <cfloat>
#include <cmath>
#include <random>

using std::minstd_rand;
using std::random_device;
using std::uniform_real_distribution;

typedef struct {
  size_t T;             // Number of iterations
  size_t *D_bar_stats;  // Number of times each threshold was selected
  double *lambda_hist;  // History of lambda values
  double *D_bar;        // History of D_bar values
  double *I_alpha_bar;  // History of time-averaged I_alpha values
  double *err_bar;      // History of time-averaged error values
} GEFAIR_RESULT;

inline size_t find_threshold_idx(double *I_alpha_cache, double *err_cache,
                                 size_t thr_candidates_size,
                                 double *thr_candidates, double lambda,
                                 double gamma) {
  // Find threshold for a given lambda value (the "oracle")
  double thri_of_lambda = 0.0;
  double min_L_value = DBL_MAX;
  for (size_t i = 0; i < thr_candidates_size; i += 1) {
    double L_value = err_cache[i] + lambda * (I_alpha_cache[i] - gamma);
    if (L_value < min_L_value) {
      min_L_value = L_value;
      thri_of_lambda = i;
    }
  }
  return thri_of_lambda;
}

inline double get_Iup(double alpha, double c, double a) {
  // Calculate Iup (reference: section 5 in the paper)
  double ca = (c + a) / (c - a);
  if (alpha == 0)
    return log(ca);
  else if (alpha == 1)
    return ca * log(ca);
  else
    return (pow(ca, alpha) - 1) / abs(alpha * (alpha - 1));
}

inline GEFAIR_RESULT *solve_gefair_loop_traced(
    size_t T, size_t thr_candidates_size, double *thr_candidates,
    double *I_alpha_cache, double *err_cache, double lambda_max,
    double *w0_mult_cache, double *w1_mult_cache, size_t lambda_0_thri,
    size_t lambda_1_thri) {
  // This function is the same as solve_gefair_loop, except that it also
  // returns the history of the hypothesis, I_alpha, and err values.
  // So look for the comments in solve_gefair_loop below for more details.
  minstd_rand rng(random_device{}());
  uniform_real_distribution<double> dist(0.0, 1.0);

  double w0 = 1.0;
  double w1 = 1.0;

  double lambda_0 = 0.0;
  double lambda_1 = lambda_max;

  size_t *D_bar_stats = (size_t *)calloc(thr_candidates_size, sizeof(size_t));
  double *lambda_hist = (double *)malloc(T * sizeof(double));
  double *D_bar = (double *)malloc(T * sizeof(double));
  double *I_alpha_bar = (double *)malloc(T * sizeof(double));
  double *err_bar = (double *)malloc(T * sizeof(double));

  double D_hist_sum = 0.0;
  double I_alpha_hist_sum = 0.0;
  double err_hist_sum = 0.0;
  for (size_t t = 0; t < T; t += 1) {
    int w0_selected = int(dist(rng) < (w0 / (w0 + w1)));
    double lambda_t = w0_selected * lambda_0 + (1 - w0_selected) * lambda_1;

    size_t thri_t =
        w0_selected * lambda_0_thri + (1 - w0_selected) * lambda_1_thri;

    w0 = w0 * w0_mult_cache[thri_t];
    w1 = w1 * w1_mult_cache[thri_t];

    D_bar_stats[thri_t] += 1;
    lambda_hist[t] = lambda_t;

    D_hist_sum += thr_candidates[thri_t];
    I_alpha_hist_sum += I_alpha_cache[thri_t];
    err_hist_sum += err_cache[thri_t];

    D_bar[t] = D_hist_sum / (t + 1);
    I_alpha_bar[t] = I_alpha_hist_sum / (t + 1);
    err_bar[t] = err_hist_sum / (t + 1);
  }

  auto result = (GEFAIR_RESULT *)malloc(sizeof(GEFAIR_RESULT));
  result->T = T;
  result->D_bar_stats = D_bar_stats;
  result->lambda_hist = lambda_hist;
  result->D_bar = D_bar;
  result->I_alpha_bar = I_alpha_bar;
  result->err_bar = err_bar;

  return result;
}

inline GEFAIR_RESULT *solve_gefair_loop(size_t T, size_t thr_candidates_size,
                                        double lambda_max,
                                        double *w0_mult_cache,
                                        double *w1_mult_cache,
                                        size_t lambda_0_thri,
                                        size_t lambda_1_thri) {
  minstd_rand rng(random_device{}());
  uniform_real_distribution<double> dist(0.0, 1.0);

  double w0 = 1.0;
  double w1 = 1.0;

  double lambda_0 = 0.0;
  double lambda_1 = lambda_max;

  size_t *D_bar_stats = (size_t *)calloc(thr_candidates_size, sizeof(size_t));
  double *lambda_hist = (double *)malloc(T * sizeof(double));

  // Implementation hack: use the "index number" of the threshold instead of
  //                      the threshold itself throughout the algorithm
  for (size_t t = 0; t < T; t += 1) {
    // 1. Destiny chooses lambda_t
    int w0_selected = int(dist(rng) < (w0 / (w0 + w1)));
    double lambda_t = w0_selected * lambda_0 + (1 - w0_selected) * lambda_1;

    // 2. The learner chooses a hypothesis (threshold(float) in this case)
    size_t thri_t =
        w0_selected * lambda_0_thri + (1 - w0_selected) * lambda_1_thri;

    // 3. Destiny updates the weight vector (w0, w1)
    w0 = w0 * w0_mult_cache[thri_t];
    w1 = w1 * w1_mult_cache[thri_t];

    // 4. Save the hypothesis and lambda_t
    D_bar_stats[thri_t] += 1;
    lambda_hist[t] = lambda_t;
  }

  auto result = (GEFAIR_RESULT *)malloc(sizeof(GEFAIR_RESULT));
  result->T = T;
  result->D_bar_stats = D_bar_stats;
  result->lambda_hist = lambda_hist;
  result->D_bar = nullptr;
  result->I_alpha_bar = nullptr;
  result->err_bar = nullptr;

  return result;
}

extern "C" GEFAIR_RESULT *solve_gefair(size_t thr_candidates_size,
                                       double *thr_candidates,
                                       double *I_alpha_cache, double *err_cache,
                                       double alpha, double lambda_max,
                                       double nu, double c, double a,
                                       double gamma, bool collect_ge_history) {
  /*
    Solve GE-Fairness (reference: algorithm 1 at section 5 in the paper)
    Note that hypothesis is a float value in this implementation.
  */

  double A_alpha = 1 + lambda_max * (gamma + get_Iup(alpha, c, a));
  double B = gamma * lambda_max;

  size_t T = 4 * A_alpha * A_alpha * log(2) / (nu * nu);
  double kappa = nu / (2 * A_alpha);

  // Implementation hack: avoid repeated calculation of multiplicative factors
  double *w0_mult_cache =
      (double *)malloc(thr_candidates_size * sizeof(double));
  double *w1_mult_cache =
      (double *)malloc(thr_candidates_size * sizeof(double));
  for (size_t i = 0; i < thr_candidates_size; i += 1) {
    w0_mult_cache[i] = pow(kappa + 1.0, (err_cache[i] + B) / A_alpha);
    w1_mult_cache[i] =
        pow(kappa + 1.0,
            ((err_cache[i] + lambda_max * (I_alpha_cache[i] - gamma)) + B) /
                A_alpha);
  }

  size_t lambda_0_thri =
      find_threshold_idx(I_alpha_cache, err_cache, thr_candidates_size,
                         thr_candidates, 0.0, gamma);
  size_t lambda_1_thri =
      find_threshold_idx(I_alpha_cache, err_cache, thr_candidates_size,
                         thr_candidates, lambda_max, gamma);

  GEFAIR_RESULT *result;
  if (collect_ge_history) {
    result = solve_gefair_loop_traced(
        T, thr_candidates_size, thr_candidates, I_alpha_cache, err_cache,
        lambda_max, w0_mult_cache, w1_mult_cache, lambda_0_thri, lambda_1_thri);
  } else {
    result =
        solve_gefair_loop(T, thr_candidates_size, lambda_max, w0_mult_cache,
                          w1_mult_cache, lambda_0_thri, lambda_1_thri);
  }

  free(w0_mult_cache);
  free(w1_mult_cache);

  return result;
}

extern "C" void free_gefair_result(GEFAIR_RESULT *result) {
  if (result->I_alpha_bar != nullptr) free(result->I_alpha_bar);
  if (result->err_bar != nullptr) free(result->err_bar);
  if (result->D_bar != nullptr) free(result->D_bar);
  free(result->D_bar_stats);
  free(result->lambda_hist);
  free(result);
}
