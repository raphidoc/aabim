functions {
  // Returns [n_wl, 2] where col 1 = a(λ), col 2 = bb(λ)
  matrix iop_from_oac_spm(
    vector wavelength,
    vector a_w,
    vector a0,
    vector a1,
    vector bb_w,
    real chl,
    real a_g_440,
    real spm,
    real a_nap_star,
    real bb_p_star,
    real a_g_s,
    real a_nap_s,
    real bb_p_gamma
    );

    vector forward_am03_ad(
      vector wavelength,
      vector a,
      vector bb,
      int water_type,
      real theta_sun_deg,
      real theta_view_deg,
      int shallow,
      real h_w,
      vector r_b
      );

      real mvn_lowrank_orth_corr_lpdf(
        vector y,
        vector mu,
        matrix U,              // [n, q], orthonormal columns
        matrix L,              // [q, q], lower Cholesky of Sigma in PC space
        real sigma2
        ) {
          int n = num_elements(y);
          int q = cols(U);
          vector[n] r = y - mu;

          vector[q] a = U' * r;                 // PC coords
          vector[n] e = r - U * a;              // orthogonal residual (because U orthonormal)

          // a ~ MVN(0, L L')
          vector[q] z = mdivide_left_tri_low(L, a);
          real lp_a = -0.5 * (q * log(2*pi()) + 2 * sum(log(diagonal(L))) + dot_self(z));

          // e ~ N(0, sigma2 I) in the (n-q)-dim orthogonal complement
          real lp_e = -0.5 * ((n - q) * log(2*pi()) + (n - q) * log(sigma2) + dot_self(e) / sigma2);

          return lp_a + lp_e;
        }


}

data {
  int<lower=1> n_wl;
  vector[n_wl] wavelength;
  vector[n_wl] rrs_obs;
  vector<lower=1e-8>[n_wl] sigma_rrs;

  // LUTs on same wavelength grid
  vector[n_wl] a_w;
  vector[n_wl] a0;
  vector[n_wl] a1;
  vector[n_wl] bb_w;

  // ----- Bottom prior: class-conditional GMM on mean-normalized shape -----
  int<lower=1> K;
  int<lower=1> q;

  array[K] vector[n_wl] eta_mu_k;
  matrix[n_wl, q] eta_U;
  array[K] matrix[q, q] eta_L_k;
  vector<lower=0>[K] eta_sigma2_k;
  simplex[K] eta_pi;

  // Geometry / flags
  int<lower=1,upper=2> water_type;
  real theta_sun;
  real theta_view;
  int<lower=0,upper=1> shallow;

  // prior hyperparameters
  real a_nap_star_mu;
  real a_nap_star_sd;
  real bb_p_star_mu;
  real bb_p_star_sd;

  real a_g_s_mu;
  real a_g_s_sd;
  real a_nap_s_mu;
  real a_nap_s_sd;
  real bb_p_gamma_mu;
  real bb_p_gamma_sd;

  // Depth prior
  real h_w_mu;
  real h_w_sd;
}

parameters {
  // Water / IOP parameters
  real<lower=0> chl;
  real<lower=0> a_g_440;
  real<lower=0> spm;

  real<lower=0> a_nap_star;
  real<lower=0> bb_p_star;

  real<lower=0> a_g_s;
  real<lower=0> a_nap_s;
  real<lower=0> bb_p_gamma;

  real<lower=0, upper=30> h_w;

  // Bottom: channelwise raw spectrum
  vector[n_wl] eta_b;

  // Model discrepancy (same units as Rrs)
  real<lower=0> sigma_model;
}

transformed parameters {
  vector[n_wl] a;
  vector[n_wl] bb;

  vector[n_wl] r_b_norm;
  vector[n_wl] r_b;

  vector[n_wl] rrs_model;
  vector[n_wl] sigma_tot;

  // IOPs (autodiff)
  {
    matrix[n_wl, 6] iop = iop_from_oac_spm(
      wavelength, a_w, a0, a1, bb_w,
      chl,
      a_g_440,
      spm,
      a_nap_star,
      bb_p_star,
      a_g_s,
      a_nap_s,
      bb_p_gamma
    );
    a  = iop[, 1];
    bb = iop[, 2];
  }

  // r_b = r_b_raw;
  r_b = inv_logit(eta_b);

  // Forward model
  rrs_model = forward_am03_ad(
    wavelength, a, bb,
    water_type, theta_sun, theta_view,
    shallow, h_w, r_b
  );

  // Total per-band uncertainty
  sigma_tot = sqrt(square(sigma_rrs) + square(sigma_model));
}

model {
  chl      ~ lognormal(log(0.9), 0.6);
  a_g_440  ~ lognormal(log(0.25), 0.6);
  spm      ~ lognormal(log(2), 1.6);

  a_nap_star  ~ normal(a_nap_star_mu, a_nap_star_sd);
  bb_p_star   ~ normal(bb_p_star_mu, bb_p_star_sd);

  a_g_s       ~ normal(a_g_s_mu, a_g_s_sd);
  a_nap_s     ~ normal(a_nap_s_mu, a_nap_s_sd);
  bb_p_gamma  ~ normal(bb_p_gamma_mu, bb_p_gamma_sd);

  h_w ~ normal(h_w_mu, h_w_sd);

  {
      {
        vector[K] log_comp;
        for (k in 1:K) {
          log_comp[k] =
          log(eta_pi[k]) +
          mvn_lowrank_orth_corr_lpdf(eta_b | eta_mu_k[k], eta_U, eta_L_k[k], eta_sigma2_k[k]);
        }
        target += log_sum_exp(log_comp);
      }
  }

  // Model discrepancy
  sigma_model ~ normal(0, 0.00005);

  // ----- Likelihood -----
  rrs_obs ~ normal(rrs_model, sigma_tot);
}

generated quantities {
  vector[n_wl] r_b_hat = r_b;
  vector[n_wl] rrs_hat = rrs_model;
}
