// Aquatic-only inversion model.
//
// Input:  rho_w = π × Rrs  (water-leaving reflectance, already atmospherically
//         corrected).  No atmospheric parameters are estimated.
//
// Forward model:  AM03 RTM (Albert & Mobley 2003) → Rrs → ρ_w = π × Rrs.
//
// Parameters
// ----------
// chl        Chlorophyll-a concentration (mg m⁻³)
// a_g_440    CDOM + NAP absorption at 440 nm (m⁻¹)
// spm        Suspended particulate matter (g m⁻³)
// bb_p_gamma Particulate backscattering spectral slope
// a_g_slope  CDOM spectral slope (nm⁻¹)
// h_w        Water depth (m)         – active only when shallow = 1
// rb_logit   Spectral bottom reflectance in logit space (n_wl)

functions {
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
    int    water_type,
    real   theta_sun_deg,
    real   theta_view_deg,
    int    shallow,
    real   h_w,
    vector r_b
  );
}

data {
  int<lower=1> n_wl;
  vector[n_wl] wavelength;

  // Observed water-leaving reflectance (ρ_w = π × Rrs) and per-band uncertainty
  vector[n_wl]          rho_obs;
  vector<lower=0>[n_wl] rho_sigma;

  // Viewing geometry (per-segment median)
  real theta_sun_deg;
  real theta_view_deg;
  int<lower=1,upper=2> water_type;
  int<lower=0,upper=1> shallow;

  // IOP LUT vectors (interpolated to sensor wavelengths)
  vector[n_wl] a_w;
  vector[n_wl] a0;
  vector[n_wl] a1;
  vector[n_wl] bb_w;

  // Bottom reflectance library prior (reflectance scale, per-wavelength)
  vector<lower=1e-6, upper=1-1e-6>[n_wl] rb_mu;
  vector<lower=1e-6>[n_wl]               rb_sd;
}

parameters {
  real<lower=0>        chl;
  real<lower=0>        a_g_440;
  real<lower=0>        spm;
  real<lower=0>        bb_p_gamma;
  real<lower=0>        a_g_slope;
  real<lower=0>        h_w;
  vector[n_wl]         rb_logit;
}

transformed parameters {
  vector[n_wl] r_b = inv_logit(rb_logit);

  matrix[n_wl, 6] iop = iop_from_oac_spm(
    wavelength, a_w, a0, a1, bb_w,
    chl, a_g_440, spm,
    /* a_nap_star */ 0.0051,
    /* bb_p_star  */ 0.0047,
    a_g_slope,
    /* a_nap_s    */ 0.006,
    bb_p_gamma
  );

  vector[n_wl] a  = iop[, 1];
  vector[n_wl] bb = iop[, 2];

  // ρ_w = π × Rrs
  vector[n_wl] rho_hat = pi() * forward_am03_ad(
    wavelength, a, bb,
    water_type, theta_sun_deg, theta_view_deg,
    shallow, h_w, r_b
  );
}

model {
  // Priors
  chl        ~ lognormal(0.0,  1.0);
  a_g_440    ~ lognormal(-1.0, 1.0);
  spm        ~ lognormal(0.0,  1.0);
  bb_p_gamma ~ normal(0.7,  0.3);
  a_g_slope  ~ normal(0.015, 0.01);
  h_w        ~ normal(3.0,  3.0);

  // Bottom reflectance prior in logit space (delta-method approximation)
  {
    vector[n_wl] muL;
    vector[n_wl] sdL;
    for (i in 1:n_wl) {
      muL[i] = logit(rb_mu[i]);
      sdL[i] = fmax(rb_sd[i] / (rb_mu[i] * (1.0 - rb_mu[i]) + 1e-8), 1e-3);
    }
    rb_logit ~ normal(muL, sdL);
  }

  // Likelihood
  rho_obs ~ normal(rho_hat, rho_sigma);
}
