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


  vector forward_sensor_reflectance_raw_lut(
    vector wavelength,
    vector a,
    vector bb,
    int water_type,
    real theta_sun_deg,
    real theta_view_deg,
    int shallow,
    real h_w,
    vector r_b,
    vector aod_grid,
    matrix rho_path_ra,
    matrix t_ra,
    matrix s_ra,
    real aod550
  );
}


data {
  int<lower=1> n_wl;
  vector[n_wl] wavelength;

  // Observations: at-sensor reflectance per segment (median) and uncertainty (MAD or sigma)
  vector[n_wl] rho_obs;
  vector<lower=0>[n_wl] rho_sigma;

  // Fixed geometry / meteo per segment
  real theta_sun_deg;
  real theta_view_deg;
  int<lower=1,upper=2> water_type;

  // Shallow switch: if 0, r_b ignored; if 1, used
  int<lower=0,upper=1> shallow;

  // Atmospheric LUT already sliced at fixed geometry/meteo, only AOD axis remains
  int<lower=2> Naod;
  vector[Naod] aod_grid;
  matrix[Naod, n_wl] rho_path_ra;
  matrix[Naod, n_wl] t_ra;
  matrix[Naod, n_wl] s_ra;

  // Water IOP LUT inputs (your existing a_w, a0, a1, bb_w etc.)
  vector[n_wl] a_w;
  vector[n_wl] a0;
  vector[n_wl] a1;
  vector[n_wl] bb_w;

  // Prior for spectral bottom reflectance from library:
  // Provide mu and sd on reflectance scale in (0,1). We will map to logit space.
  vector<lower=1e-6,upper=1-1e-6>[n_wl] rb_mu;
  vector<lower=1e-6>[n_wl] rb_sd;

  // Optional: depth prior inputs if you want
  // real<lower=0> hw_prior_mu;
  // real<lower=0> hw_prior_sd;
}

parameters {
  // Atmosphere
  real<lower=0> aod550;

  // Aquatic state (as you specified)
  real<lower=0> chl;
  real<lower=0> a_g_440;
  real<lower=0> spm;
  real<lower=0> bb_p_gamma;
  real<lower=0> a_g_slope;

  // Depth
  real<lower=0> h_w;

  // Spectral bottom reflectance in logit space (unconstrained)
  vector[n_wl] rb_logit;
}

transformed parameters {
  // Bottom reflectance on (0,1)
  vector[n_wl] r_b = inv_logit(rb_logit);

  // IOPs from your existing core (SPM formulation)
  // returns [n_wl,6] with columns:
  // (a, bb, a_phy, a_g, a_nap, bb_p)
  matrix[n_wl, 6] iop = iop_from_oac_spm(
    wavelength, a_w, a0, a1, bb_w,
    chl, a_g_440, spm,
    /* a_nap_star */ 0.0051,      // <- if you want it in state, add param + wire here
    /* bb_p_star  */ 0.0047,      // <- same
    a_g_slope,
    /* a_nap_s    */ 0.006,     // <- same
    bb_p_gamma
  );

  vector[n_wl] a  = iop[,1];
  vector[n_wl] bb = iop[,2];

  // Forward model: rho_at_sensor predicted
  vector[n_wl] rho_hat = forward_sensor_reflectance_raw_lut(
    wavelength, a, bb,
    water_type,
    theta_sun_deg, theta_view_deg,
    shallow,
    h_w,
    r_b,
    aod_grid, rho_path_ra, t_ra, s_ra,
    aod550
  );
}

model {
  // ---- Priors ----
  aod550      ~ normal(0.10, 0.6);
  chl         ~ normal(1.0,  0.8);
  a_g_440     ~ normal(0.6, 1);
  spm         ~ normal(1.0,  0.9);
  bb_p_gamma  ~ normal(0.7, 0.3);
  a_g_slope   ~ normal(0.015, 0.01);

  h_w         ~ normal(3, 3);

  // Spectral bottom prior in logit space:
  // If rb_mu, rb_sd are per-wavelength mean/std on reflectance scale,
  // approximate logit-space std by delta method:
  //   sd_logit ≈ sd / (mu*(1-mu))
  {
    vector[n_wl] muL;
    vector[n_wl] sdL;
    for (i in 1:n_wl) {
      muL[i] = logit(rb_mu[i]);
      sdL[i] = rb_sd[i] / (rb_mu[i] * (1.0 - rb_mu[i]) + 1e-8);
      sdL[i] = fmax(sdL[i], 1e-3);
    }
    rb_logit ~ normal(muL, sdL);
  }

  // ---- Likelihood ----
  rho_obs ~ normal(rho_hat, rho_sigma);
}
