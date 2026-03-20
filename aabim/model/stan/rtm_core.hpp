#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace saber {

// default: works for double
inline double value(double x) { return x; }

#ifdef STAN_MATH_HPP
template <typename T>
inline double value(const T& x) { return stan::math::value_of(x); }
#endif

// geometry helper stays double-only (data)
inline void snell_law_double(double theta_view_deg, double theta_sun_deg,
                             double* view_w_rad, double* sun_w_rad) {
  constexpr double deg2rad = 3.14159265358979323846 / 180.0;
  constexpr double n_air = 1.0;
  constexpr double n_water = 1.34;
  const double tv = theta_view_deg * deg2rad;
  const double ts = theta_sun_deg  * deg2rad;

  const double sin_tv_w = (n_air / n_water) * std::sin(tv);
  const double sin_ts_w = (n_air / n_water) * std::sin(ts);

  const double stv = std::max(-1.0, std::min(1.0, sin_tv_w));
  const double sts = std::max(-1.0, std::min(1.0, sin_ts_w));
  *view_w_rad = std::asin(stv);
  *sun_w_rad  = std::asin(sts);
}

// Returns [n_wl,2] col1=a, col2=bb
template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 2>
iop_from_oac_core(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                      const Eigen::Ref<const Eigen::VectorXd>& a_w,
                      const Eigen::Ref<const Eigen::VectorXd>& a0,
                      const Eigen::Ref<const Eigen::VectorXd>& a1,
                      const Eigen::Ref<const Eigen::VectorXd>& bb_w,
                      const T& chl,
                      const T& a_gnap_440,
                      const T& bb_p_550,
                      const T& a_gnap_s,
                      const T& bb_p_gamma) {
  const int n = wavelength.size();
  if (a_w.size()!=n || a0.size()!=n || a1.size()!=n || bb_w.size()!=n)
    throw std::domain_error("iop_from_oac_core: LUT size mismatch vs wavelength");

  Eigen::Matrix<T, Eigen::Dynamic, 2> out(n, 2);

  // IMPORTANT: call exp/log/pow unqualified (ADL picks Stan overloads when T is Stan)
  const T aph_440 = T(0.06) * pow(chl, T(0.65));

  for (int i = 0; i < n; ++i) {
    const T wl = T(wavelength(i));

    T a_phy = (T(a0(i)) + T(a1(i)) * log(aph_440)) * aph_440;
    if (saber::value(a_phy) < 0.0) a_phy = T(0.0);

    const T a_gnap = a_gnap_440 * exp(-a_gnap_s * (wl - T(440.0)));
    const T bb_p   = bb_p_550   * pow(wl / T(550.0), -bb_p_gamma);

    out(i, 0) = T(a_w(i))  + a_phy + a_gnap;
    out(i, 1) = T(bb_w(i)) + bb_p;
  }
  return out;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 6>
iop_from_oac_spm_core(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                  const Eigen::Ref<const Eigen::VectorXd>& a_w,
                  const Eigen::Ref<const Eigen::VectorXd>& a0,
                  const Eigen::Ref<const Eigen::VectorXd>& a1,
                  const Eigen::Ref<const Eigen::VectorXd>& bb_w,
                  const T& chl,
                  const T& a_g_440,
                  const T& spm,
                  const T& a_nap_star,
                  const T& bb_p_star,
                  const T& a_g_s,
                  const T& a_nap_s,
                  const T& bb_p_gamma) {
  const int n = wavelength.size();
  if (a_w.size()!=n || a0.size()!=n || a1.size()!=n || bb_w.size()!=n)
    throw std::domain_error("iop_from_oac_core: LUT size mismatch vs wavelength");

  Eigen::Matrix<T, Eigen::Dynamic, 6> out(n, 6);

  // IMPORTANT: call exp/log/pow unqualified (ADL picks Stan overloads when T is Stan)
  const T aph_440 = T(0.06) * pow(chl, T(0.65));

  for (int i = 0; i < n; ++i) {
    const T wl = T(wavelength(i));

    T a_phy = (T(a0(i)) + T(a1(i)) * log(aph_440)) * aph_440;
    if (saber::value(a_phy) < 0.0) a_phy = T(0.0);

    const T a_g = a_g_440 * exp(-a_g_s * (wl - T(440.0)));

    const T a_nap_440 = a_nap_star * spm;
    const T bb_p_550 = bb_p_star * spm;

    const T a_nap = a_nap_440 * exp(-a_nap_s * (wl - T(440.0)));
    const T bb_p   = bb_p_550 * pow(wl / T(550.0), -bb_p_gamma);

    out(i, 0) = T(a_w(i))  + a_phy + a_g + a_nap;
    out(i, 1) = T(bb_w(i)) + bb_p;
    out(i, 2) = a_phy;
    out(i, 3) = a_g;
    out(i, 4) = a_nap;
    out(i, 5) = bb_p;
  }
  return out;
}

template <typename T, typename DerivedRB>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
forward_am03_core(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& bb,
                  int water_type,
                  double theta_sun_deg,
                  double theta_view_deg,
                  int shallow,
                  const T& h_w,
                  const Eigen::MatrixBase<DerivedRB>& r_b) {
  const int n = wavelength.size();
  if (a.size()!=n || bb.size()!=n)
    throw std::domain_error("forward_am03_core: size mismatch (a/bb vs wavelength)");
  if (shallow && r_b.size()!=n)
    throw std::domain_error("forward_am03_core: r_b size mismatch for shallow water");

  double view_w_rad = 0.0, sun_w_rad = 0.0;
  snell_law_double(theta_view_deg, theta_sun_deg, &view_w_rad, &sun_w_rad);
  const double cos_sun  = std::cos(sun_w_rad);
  const double cos_view = std::cos(view_w_rad);
  if (cos_sun <= 0.0 || cos_view <= 0.0)
    throw std::domain_error("forward_am03_core: invalid geometry (cos <= 0)");

  Eigen::Matrix<T, Eigen::Dynamic, 1> rrs(n);

  for (int i = 0; i < n; ++i) {
    const T ext = a(i) + bb(i);
    if (saber::value(ext) <= 0.0) { rrs(i) = T(0.0); continue; }

    const T omega_b = bb(i) / ext;

    T f_rs;
    if (water_type == 1) {
      f_rs = T(0.095);
    } else if (water_type == 2) {
      f_rs = T(0.0512)
      * (T(1.0) + T(4.6659)*omega_b - T(7.8387)*omega_b*omega_b + T(5.4571)*omega_b*omega_b*omega_b)
      * (T(1.0) + T(0.1098)/cos_sun)
      * (T(1.0) + T(0.4021)/cos_view);
    } else {
      throw std::domain_error("forward_am03_core: water_type must be 1 or 2");
    }

    const T rrs_deep = f_rs * omega_b;

    if (shallow) {
      const double k0 = (water_type == 1) ? 1.0395 : 1.0546;

      const T Kd  = T(k0) * (ext / cos_sun);
      const T kuW = (ext / cos_view) * pow(T(1.0) + omega_b, T(3.5421)) * (T(1.0) - T(0.2786) / cos_sun);
      const T kuB = (ext / cos_view) * pow(T(1.0) + omega_b, T(2.2658)) * (T(1.0) - T(0.0577) / cos_sun);

      const T Ars1 = T(1.1576);
      const T Ars2 = T(1.0389);

      rrs(i) = rrs_deep * (T(1.0) - Ars1 * exp(-h_w * (Kd + kuW)))
        + Ars2 * (r_b(i) / T(M_PI)) * exp(-h_w * (Kd + kuB));
    } else {
      rrs(i) = rrs_deep;
    }

    if (!std::isfinite(saber::value(rrs(i)))) rrs(i) = T(0.0);
  }

  return rrs;
}

template <typename T, typename DerivedRRS>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
solve_rb_am03_core(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                   const Eigen::Matrix<T, Eigen::Dynamic, 1>& a,
                   const Eigen::Matrix<T, Eigen::Dynamic, 1>& bb,
                   int water_type,
                   double theta_sun_deg,
                   double theta_view_deg,
                   const T& h_w,
                   const Eigen::MatrixBase<DerivedRRS>& rrs_obs) {
  const int n = wavelength.size();
  if (a.size() != n || bb.size() != n)
    throw std::domain_error("solve_rb_am03_core: size mismatch (a/bb vs wavelength)");
  if (rrs_obs.size() != n)
    throw std::domain_error("solve_rb_am03_core: rrs_obs size mismatch vs wavelength");

  double view_w_rad = 0.0, sun_w_rad = 0.0;
  snell_law_double(theta_view_deg, theta_sun_deg, &view_w_rad, &sun_w_rad);
  const double cos_sun  = std::cos(sun_w_rad);
  const double cos_view = std::cos(view_w_rad);
  if (cos_sun <= 0.0 || cos_view <= 0.0)
    throw std::domain_error("solve_rb_am03_core: invalid geometry (cos <= 0)");

  Eigen::Matrix<T, Eigen::Dynamic, 1> r_b(n);

  const T Ars1 = T(1.1576);
  const T Ars2 = T(1.0389);

  for (int i = 0; i < n; ++i) {
    const T ext = a(i) + bb(i);
    if (saber::value(ext) <= 0.0) { r_b(i) = T(0.0); continue; }

    const T omega_b = bb(i) / ext;

    T f_rs;
    if (water_type == 1) {
      f_rs = T(0.095);
    } else if (water_type == 2) {
      f_rs = T(0.0512)
      * (T(1.0) + T(4.6659)*omega_b - T(7.8387)*omega_b*omega_b + T(5.4571)*omega_b*omega_b*omega_b)
      * (T(1.0) + T(0.1098)/cos_sun)
      * (T(1.0) + T(0.4021)/cos_view);
    } else {
      throw std::domain_error("solve_rb_am03_core: water_type must be 1 or 2");
    }

    const T rrs_deep = f_rs * omega_b;

    const double k0 = (water_type == 1) ? 1.0395 : 1.0546;

    const T Kd  = T(k0) * (ext / cos_sun);
    const T kuW = (ext / cos_view) * pow(T(1.0) + omega_b, T(3.5421)) * (T(1.0) - T(0.2786) / cos_sun);
    const T kuB = (ext / cos_view) * pow(T(1.0) + omega_b, T(2.2658)) * (T(1.0) - T(0.0577) / cos_sun);

    const T termW = exp(-h_w * (Kd + kuW));
    const T termB = exp(-h_w * (Kd + kuB));

    // water-column-only contribution (no bottom)
    const T rrs_wc = rrs_deep * (T(1.0) - Ars1 * termW);

    // Solve for r_b:
    // rrs_obs = rrs_wc + Ars2 * (r_b/pi) * termB
    // => r_b = (pi/Ars2) * (rrs_obs - rrs_wc) / termB
    //        = (pi/Ars2) * (rrs_obs - rrs_wc) * exp(+h_w*(Kd+kuB))
    const T delta = T(rrs_obs(i)) - rrs_wc;

    // If termB is ~0, inversion explodes; guard numerically
    if (saber::value(termB) <= 0.0) { r_b(i) = T(0.0); continue; }

    r_b(i) = (T(M_PI) / Ars2) * delta / termB;

    if (!std::isfinite(saber::value(r_b(i)))) r_b(i) = T(0.0);
  }

  return r_b;
}


} // namespace saber
