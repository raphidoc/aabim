// atm_core.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

#include "rtm_core.hpp"

namespace saber {

struct AtmLut1D {
  Eigen::VectorXd aod_grid;              // size Na
  Eigen::MatrixXd rho_path_ra;           // [Na, Nwl]
  Eigen::MatrixXd t_ra;                  // [Na, Nwl]
  Eigen::MatrixXd s_ra;                  // [Na, Nwl]
};

inline int lower_index(const Eigen::VectorXd& g, double x) {
  if (x <= g(0)) return 0;
  const int n = g.size();
  if (x >= g(n-1)) return n-2;
  auto it = std::upper_bound(g.data(), g.data()+n, x);
  int i = int(it - g.data()) - 1;
  if (i < 0) i = 0;
  if (i > n-2) i = n-2;
  return i;
}

template <typename T>
inline T lerp(const T& a, const T& b, const T& w) { return a + w*(b-a); }

template <typename T>
inline void interp_atm_1d_aod(
    const AtmLut1D& lut,
    const T& aod550,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* rho_path_out,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* t_ra_out,
    Eigen::Matrix<T, Eigen::Dynamic, 1>* s_ra_out) {

  const int Na  = lut.aod_grid.size();
  const int Nwl = lut.rho_path_ra.cols();
  if (lut.rho_path_ra.rows()!=Na || lut.t_ra.rows()!=Na || lut.s_ra.rows()!=Na)
    throw std::domain_error("interp_atm_1d_aod: LUT rows mismatch vs aod_grid");

  const double aod_d = saber::value(aod550);          // uses your value() in rtm_core.hpp
  const int i0 = lower_index(lut.aod_grid, aod_d);
  const int i1 = i0 + 1;

  const double g0 = lut.aod_grid(i0);
  const double g1 = lut.aod_grid(i1);
  const double denom = (g1 - g0);
  const double w_d = (denom > 0.0) ? (aod_d - g0)/denom : 0.0;

  const T w = T(std::max(0.0, std::min(1.0, w_d)));

  rho_path_out->resize(Nwl);
  t_ra_out->resize(Nwl);
  s_ra_out->resize(Nwl);

  for (int j=0; j<Nwl; ++j) {
    const T rp0 = T(lut.rho_path_ra(i0,j)), rp1 = T(lut.rho_path_ra(i1,j));
    const T tr0 = T(lut.t_ra(i0,j)),        tr1 = T(lut.t_ra(i1,j));
    const T sr0 = T(lut.s_ra(i0,j)),        sr1 = T(lut.s_ra(i1,j));

    (*rho_path_out)(j) = lerp(rp0, rp1, w);
    (*t_ra_out)(j)     = lerp(tr0, tr1, w);
    (*s_ra_out)(j)     = lerp(sr0, sr1, w);
  }
}

// Convert rrs(0-) -> Rrs(0+) (Lee et al. 1998)
template <typename T>
inline T rrs_0m_to_0p_core(const T& rrs_0m) {
  // rrs_0p = 0.518 * rrs_0m / (1 - 1.562 * rrs_0m)
  const T denom = T(1.0) - T(1.562) * rrs_0m;
  if (saber::value(denom) == 0.0) return T(0.0);
  return T(0.518) * rrs_0m / denom;
}

// Combined forward:
//  1) AM03 gives rrs(0-) (subsurface)
//  2) convert to Rrs(0+)
//  3) rho_s = pi * Rrs(0+)
//  4) rho_sensor = rho_path + t_ra * rho_s / (1 - s_ra * rho_s)
template <typename T, typename DerivedRB>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
forward_sensor_reflectance_core(
  const Eigen::Ref<const Eigen::VectorXd>& wavelength,
  // aquatic IOPs
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& bb,
  // AM03 controls
  int water_type,
  double theta_sun_deg,
  double theta_view_deg,
  int shallow,
  const T& h_w,
  const Eigen::MatrixBase<DerivedRB>& r_b,
  // atmosphere (1D AOD LUT already sliced on geometry/meteo)
  const AtmLut1D& atm_lut,
  const T& aod550
) {
  const int n = wavelength.size();
  if (a.size() != n || bb.size() != n)
    throw std::domain_error("forward_sensor_reflectance_core: size mismatch (a/bb vs wavelength)");
  if (shallow && r_b.size() != n)
    throw std::domain_error("forward_sensor_reflectance_core: r_b size mismatch for shallow water");

  // ---- 1) aquatic forward: rrs(0-) from AM03 ----
  const Eigen::Matrix<T, Eigen::Dynamic, 1> rrs_0m =
    saber::forward_am03_core<T>(wavelength, a, bb,
                                water_type, theta_sun_deg, theta_view_deg,
                                shallow, h_w, r_b.derived());

  // ---- 2) convert to Rrs(0+) and rho_s ----
  Eigen::Matrix<T, Eigen::Dynamic, 1> rho_s(n);
  for (int i = 0; i < n; ++i) {
    T Rrs_0p = rrs_0m_to_0p_core(rrs_0m(i));
    if (!std::isfinite(saber::value(Rrs_0p))) Rrs_0p = T(0.0);
    if (saber::value(Rrs_0p) < 0.0) Rrs_0p = T(0.0);          // keep physical
    rho_s(i) = T(M_PI) * Rrs_0p;
  }

  // ---- 3) interpolate atmospheric terms along AOD ----
  Eigen::Matrix<T, Eigen::Dynamic, 1> rho_path, t_ra, s_ra;
  saber::interp_atm_1d_aod<T>(atm_lut, aod550, &rho_path, &t_ra, &s_ra);

  if (rho_path.size() != n || t_ra.size() != n || s_ra.size() != n)
    throw std::domain_error("forward_sensor_reflectance_core: atm terms size mismatch vs wavelength");

  // ---- 4) combine to sensor reflectance ----
  Eigen::Matrix<T, Eigen::Dynamic, 1> rho_sensor(n);

  for (int i = 0; i < n; ++i) {
    // rho = rho_path + t_ra * rho_s / (1 - s_ra * rho_s)
    const T denom = T(1.0) - s_ra(i) * rho_s(i);

    // guard: if denom too small/negative, clamp to avoid blow-ups
    double denom_d = saber::value(denom);
    if (!std::isfinite(denom_d) || denom_d < 1e-6) {
      rho_sensor(i) = rho_path(i); // fall back to path term only
      continue;
    }

    rho_sensor(i) = rho_path(i) + (t_ra(i) * rho_s(i)) / denom;

    if (!std::isfinite(saber::value(rho_sensor(i)))) rho_sensor(i) = rho_path(i);
  }

  return rho_sensor;
}

} // namespace saber
