#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Set state dimension
  n_x_ = 5;

  // Set augmented dimension
  n_aug_ = 7;

  // Define spreading parameter
  lambda_ = 3 - n_aug_;

  // Set laser and radar measurement dimension
  las_n_z_ = 2;
  rad_n_z_ = 3;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Create example matrix with predicted sigma points in state space
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  sum_lanmda_n_aug_ = lambda_ + n_aug_;
  sqrt_sum_lanmda_n_aug_ = sqrt(sum_lanmda_n_aug_);

  // Set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / sum_lanmda_n_aug_;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5 / sum_lanmda_n_aug_;
  }

  // Define measurement noise covariance matrix
  las_R_ = MatrixXd(las_n_z_, las_n_z_);
  las_R_ << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  rad_R_ = MatrixXd(rad_n_z_, rad_n_z_);
  rad_R_ << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  std_a_square_ = std_a_ * std_a_;
  std_yawdd_square_ = std_yawdd_ * std_yawdd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      const double ro = meas_package.raw_measurements_[0];
      const double theta = meas_package.raw_measurements_[1];
      const double ro_dot = meas_package.raw_measurements_[2];

      // TODO: Init and tune the x_ and P_
      x_ << ro * cos(theta), ro * sin(theta), ro_dot * cos(theta), 0, 0;
      P_ << 1, 1, 0, 0, 0,
          1, 1, 0, 0, 0,
          0, 0, 10, 0, 0,
          0, 0, 0, std_radphi_ * std_radphi_, 0,
          0, 0, 0, 0, 1;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Init and tune the x_ and P_
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
          0, std_laspy_ * std_laspy_, 0, 0, 0,
          0, 0, 10, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */
  const float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0f;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  // cout << "After prediction x_:" << endl << x_ << endl << endl;
  // cout << "After prediction P_:" << endl << P_ << endl << endl;

  /**
   * Update
   */
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
    // const double ro = meas_package.raw_measurements_[0];
    // const double theta = meas_package.raw_measurements_[1];
    // cout << "Input measurement:" << endl << ro * cos(theta) << endl << ro * sin(theta) << endl << endl;
  } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
    // cout << "Input measurement:" << endl << meas_package.raw_measurements_ << endl << endl;
  }

  // cout << "x_:" << endl << x_ << endl << endl;
  // cout << "P_:" << endl << P_ << endl << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints();
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

void UKF::BaseUpdate(int n_z, VectorXd z, MatrixXd Zsig, MatrixXd R) {
  // Calculate mean predicted measurement
  const VectorXd z_pred = Zsig * weights_;

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);

  // Calculate measurement covariance matrix S and cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  S += R;

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  x_ = x_ + K * (z - z_pred);
  P_ = P_ - K * S * K.transpose();

  // cout << "Input measurement:" << endl << z << endl << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(las_n_z_, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);

    Zsig.col(i) << px, py;
  }

  BaseUpdate(las_n_z_, meas_package.raw_measurements_, Zsig, las_R_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(rad_n_z_, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);

    if (px != 0 || py != 0) {
      const double sss = sqrt(px * px + py * py);
      Zsig.col(i) << sss, atan2(py, px), (px * cos(yaw) * v + py * sin(yaw) * v) / sss;
    } else {
      // If the car is on the zero point, use its direction for phi
      Zsig.col(i) << 0, yaw, v;
    }
  }

  BaseUpdate(rad_n_z_, meas_package.raw_measurements_, Zsig, rad_R_);
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_square_;
  P_aug(6, 6) = std_yawdd_square_;

  // Create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // Create augmented sigma points
  const MatrixXd Xdelta = sqrt_sum_lanmda_n_aug_ * A;
  const MatrixXd Xmulti = x_aug * MatrixXd::Ones(1, n_aug_);
  Xsig_aug << x_aug, Xmulti + Xdelta, Xmulti - Xdelta;

  // cout << "Xsig_aug:" << endl << Xsig_aug << endl << endl;
  return Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x = Xsig_aug.col(i);
    const double v = x(2);
    const double yaw = x(3);
    const double yawd = x(4);
    const double nu_a = x(5);
    const double nu_yawdd = x(6);

    VectorXd x_pred = VectorXd(n_x_);
    const double half_squaer_delta_t = 0.5 * delta_t * delta_t;
    const double cos_yaw = cos(yaw);
    const double sin_yaw = sin(yaw);

    if (fabs(yawd) > 0.001) {
      x_pred << v / yawd * (sin(yaw + yawd * delta_t) - sin_yaw) + half_squaer_delta_t * cos_yaw * nu_a,
          v / yawd * (-cos(yaw + yawd * delta_t) + cos_yaw) + half_squaer_delta_t * sin_yaw * nu_a,
          delta_t * nu_a,
          yawd * delta_t + half_squaer_delta_t * nu_yawdd,
          delta_t * nu_yawdd;
    } else {
      x_pred << v * cos_yaw * delta_t + half_squaer_delta_t * cos_yaw * nu_a,
          v * sin_yaw * delta_t + half_squaer_delta_t * sin_yaw * nu_a,
          delta_t * nu_a,
          yawd * delta_t + half_squaer_delta_t * nu_yawdd,
          delta_t * nu_yawdd;
    }

    Xsig_pred_.col(i) = x.head(n_x_) + x_pred;
  }
  // cout << "Xsig_pred_" << endl << Xsig_pred_ << endl << endl;
}

void UKF::PredictMeanAndCovariance() {
  // Predict state mean
  x_ = Xsig_pred_ * weights_;

  // Predict state covariance matrix
  P_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}


void NormalizeAngle(double &phi) {
  phi = atan2(sin(phi), cos(phi));
}