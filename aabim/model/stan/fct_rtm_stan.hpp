#pragma once
#include <stan/math.hpp>
#include <Eigen/Dense>
#include "../include/rtm_core.hpp"
#include "../include/atm_core.hpp"


// Keep Stan-friendly signatures (return_type_t, pstream__) but call the core.
template <typename Tchl, typename Tag440, typename Tbbp550, typename Ts, typename Tgamma>
inline Eigen::Matrix<stan::return_type_t<Tchl,Tag440,Tbbp550,Ts,Tgamma>, Eigen::Dynamic, 2>
iop_from_oac(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                 const Eigen::Ref<const Eigen::VectorXd>& a_w,
                 const Eigen::Ref<const Eigen::VectorXd>& a0,
                 const Eigen::Ref<const Eigen::VectorXd>& a1,
                 const Eigen::Ref<const Eigen::VectorXd>& bb_w,
                 const Tchl& chl,
                 const Tag440& a_gnap_440,
                 const Tbbp550& bb_p_550,
                 const Ts& a_gnap_s,
                 const Tgamma& bb_p_gamma,
                 std::ostream* pstream__) {
  using T = stan::return_type_t<Tchl,Tag440,Tbbp550,Ts,Tgamma>;
  return saber::iop_from_oac_core<T>(wavelength, a_w, a0, a1, bb_w,
                                         T(chl), T(a_gnap_440), T(bb_p_550),
                                         T(a_gnap_s), T(bb_p_gamma));
}

template <typename Tchl, typename Tag440, typename Tspm,
          typename Tnapstar, typename Tbbpstar,
          typename Tags, typename Tnaps, typename Tgamma>
inline Eigen::Matrix<
  stan::return_type_t<Tchl, Tag440, Tspm, Tnapstar, Tbbpstar, Tags, Tnaps, Tgamma>,
  Eigen::Dynamic, 6>
iop_from_oac_spm(
  const Eigen::Ref<const Eigen::VectorXd>& wavelength,
  const Eigen::Ref<const Eigen::VectorXd>& a_w,
  const Eigen::Ref<const Eigen::VectorXd>& a0,
  const Eigen::Ref<const Eigen::VectorXd>& a1,
  const Eigen::Ref<const Eigen::VectorXd>& bb_w,
  const Tchl& chl,
  const Tag440& a_g_440,
  const Tspm& spm,
  const Tnapstar& a_nap_star,
  const Tbbpstar& bb_p_star,
  const Tags& a_g_s,
  const Tnaps& a_nap_s,
  const Tgamma& bb_p_gamma,
  std::ostream* pstream__
) {
  using T = stan::return_type_t<Tchl, Tag440, Tspm, Tnapstar, Tbbpstar, Tags, Tnaps, Tgamma>;

  return saber::iop_from_oac_spm_core<T>(
    wavelength, a_w, a0, a1, bb_w,
    T(chl),
    T(a_g_440),
    T(spm),
    T(a_nap_star),
    T(bb_p_star),
    T(a_g_s),
    T(a_nap_s),
    T(bb_p_gamma)
  );
}

template <typename DerivedA, typename DerivedBB, typename DerivedRB>
inline Eigen::Matrix<typename DerivedA::Scalar, Eigen::Dynamic, 1>
forward_am03_ad(const Eigen::Ref<const Eigen::VectorXd>& wavelength,
                const Eigen::MatrixBase<DerivedA>& a,
                const Eigen::MatrixBase<DerivedBB>& bb,
                int water_type,
                double theta_sun_deg,
                double theta_view_deg,
                int shallow,
                const typename DerivedA::Scalar& h_w,
                const Eigen::MatrixBase<DerivedRB>& r_b,
                std::ostream* pstream__) {
  using T = typename DerivedA::Scalar;
  return saber::forward_am03_core<T>(wavelength,
                                     a.derived(), bb.derived(),
                                     water_type, theta_sun_deg, theta_view_deg,
                                     shallow, h_w, r_b.derived());
}

// ---------------------------
// Internal worker: takes an AtmLut1D (not directly called from Stan)
// ---------------------------
template <typename DerivedA, typename DerivedBB, typename DerivedRB, typename Taod>
inline Eigen::Matrix<
  stan::return_type_t<typename DerivedA::Scalar, typename DerivedBB::Scalar,
                      typename DerivedRB::Scalar, Taod>,
                      Eigen::Dynamic, 1>
forward_sensor_reflectance_atmlut(
  const Eigen::Ref<const Eigen::VectorXd>& wavelength,
  const Eigen::MatrixBase<DerivedA>& a,
  const Eigen::MatrixBase<DerivedBB>& bb,
  int water_type,
  double theta_sun_deg,
  double theta_view_deg,
  int shallow,
  const typename DerivedA::Scalar& h_w,
  const Eigen::MatrixBase<DerivedRB>& r_b,
  const saber::AtmLut1D& atm_lut,
  const Taod& aod550,
  std::ostream* /*pstream__*/
) {
  using T = stan::return_type_t<typename DerivedA::Scalar, typename DerivedBB::Scalar,
                                typename DerivedRB::Scalar, Taod>;

  const T h_w_T = T(h_w);
  const T aod_T = T(aod550);

  Eigen::Matrix<T, Eigen::Dynamic, 1> a_T  = a.derived().template cast<T>();
  Eigen::Matrix<T, Eigen::Dynamic, 1> bb_T = bb.derived().template cast<T>();
  Eigen::Matrix<T, Eigen::Dynamic, 1> rb_T = r_b.derived().template cast<T>();

  return saber::forward_sensor_reflectance_core<T>(
    wavelength,
    a_T,
    bb_T,
    water_type,
    theta_sun_deg,
    theta_view_deg,
    shallow,
    h_w_T,
    rb_T,
    atm_lut,
    aod_T
  );
}

// ---------------------------
// Stan-callable function (single name, no overloading):
// raw LUT pieces -> build AtmLut1D -> call internal worker
// IMPORTANT: includes std::ostream* pstream__
// ---------------------------
template <typename DerivedA, typename DerivedBB, typename DerivedRB, typename Taod>
inline Eigen::Matrix<
  stan::return_type_t<typename DerivedA::Scalar, typename DerivedBB::Scalar,
                      typename DerivedRB::Scalar, Taod>,
                      Eigen::Dynamic, 1>
forward_sensor_reflectance_raw_lut(
  const Eigen::Ref<const Eigen::VectorXd>& wavelength,
  const Eigen::MatrixBase<DerivedA>& a,
  const Eigen::MatrixBase<DerivedBB>& bb,
  int water_type,
  double theta_sun_deg,
  double theta_view_deg,
  int shallow,
  const typename DerivedA::Scalar& h_w,
  const Eigen::MatrixBase<DerivedRB>& r_b,
  const Eigen::Ref<const Eigen::VectorXd>& aod_grid,
  const Eigen::Ref<const Eigen::MatrixXd>& rho_path_ra,
  const Eigen::Ref<const Eigen::MatrixXd>& t_ra,
  const Eigen::Ref<const Eigen::MatrixXd>& s_ra,
  const Taod& aod550,
  std::ostream* pstream__
) {
  saber::AtmLut1D lut;
  lut.aod_grid    = aod_grid;
  lut.rho_path_ra = rho_path_ra;
  lut.t_ra        = t_ra;
  lut.s_ra        = s_ra;

  return forward_sensor_reflectance_atmlut(
    wavelength,
    a,
    bb,
    water_type,
    theta_sun_deg,
    theta_view_deg,
    shallow,
    h_w,
    r_b,
    lut,
    aod550,
    pstream__
  );
}
