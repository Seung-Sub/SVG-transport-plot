#include "mppi_controller/mppi_controller_ros.hpp"
#include <tf2/utils.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>

namespace mppi {

MPPIControllerROS::MPPIControllerROS() : nh_(""), private_nh_("~"), tf_listener_(tf_buffer_) {
    // 최소한의 파라미터만 로드 (환경 데이터는 사용하지 않음)
    private_nh_.param("is_simulation", is_simulation_, false);
    private_nh_.param("is_reference_path_less_mode", is_reference_path_less_mode_, true);
    private_nh_.param("control_sampling_time", control_sampling_time_, 0.1);
    private_nh_.param("is_visualize_mppi", is_visualize_mppi_, false);

    std::string control_cmd_topic;
    private_nh_.param("control_cmd_topic", control_cmd_topic, std::string("/robot/move_base/cmd_vel"));

    // MPC 파라미터 로드 (필요한 항목만)
    Params params;
    params.common.is_reference_path_less_mode = is_reference_path_less_mode_;
    private_nh_.param("common/thread_num", params.common.thread_num, 12);
    private_nh_.param("common/prediction_step_size", params.common.prediction_step_size, 10);
    private_nh_.param("common/prediction_interval", params.common.prediction_interval, 0.1);
    private_nh_.param("common/max_Vx", params.common.max_Vx, 1.0);
    private_nh_.param("common/min_Vx", params.common.min_Vx, -1.0);
    // private_nh_.param("common/max_Vy", params.common.max_Vy, 1.0);
    // private_nh_.param("common/min_Vy", params.common.min_Vy, -1.0);
    // private_nh_.param("common/max_Wz", params.common.max_Wz, 1.0);
    // private_nh_.param("common/min_Wz", params.common.min_Wz, -1.0);

    // SVG_MPPi 파라미터 로드
    private_nh_.param("svg_mppi/sample_batch_num", params.svg_mppi.sample_batch_num, 1000);
    private_nh_.param("svg_mppi/lambda", params.svg_mppi.lambda, 10.0);
    private_nh_.param("svg_mppi/alpha", params.svg_mppi.alpha, 0.1);
    private_nh_.param("svg_mppi/non_biased_sampling_rate", params.svg_mppi.non_biased_sampling_rate, 0.1);
    private_nh_.param("svg_mppi/Vx_cov", params.svg_mppi.Vx_cov, 0.5);
    // private_nh_.param("svg_mppi/Vy_cov", params.svg_mppi.Vy_cov, 0.5);
    // private_nh_.param("svg_mppi/Wz_cov", params.svg_mppi.Wz_cov, 0.5);
    private_nh_.param("svg_mppi/guide_sample_num", params.svg_mppi.guide_sample_num, 1);
    private_nh_.param("svg_mppi/grad_lambda", params.svg_mppi.grad_lambda, 3.0);
    private_nh_.param("svg_mppi/sample_num_for_grad_estimation", params.svg_mppi.sample_num_for_grad_estimation, 10);
    private_nh_.param("svg_mppi/Vx_cov_for_grad_estimation", params.svg_mppi.Vx_cov_for_grad_estimation, 0.05);
    // private_nh_.param("svg_mppi/Vy_cov_for_grad_estimation", params.svg_mppi.Vy_cov_for_grad_estimation, 0.05);
    // private_nh_.param("svg_mppi/Wz_cov_for_grad_estimation", params.svg_mppi.Wz_cov_for_grad_estimation, 0.05);
    private_nh_.param("svg_mppi/svgd_step_size", params.svg_mppi.svgd_step_size, 0.1);
    private_nh_.param("svg_mppi/num_svgd_iteration", params.svg_mppi.num_svgd_iteration, 100);
    private_nh_.param("svg_mppi/is_use_nominal_solution", params.svg_mppi.is_use_nominal_solution, true);
    private_nh_.param("svg_mppi/is_covariance_adaptation", params.svg_mppi.is_covariance_adaptation, true);
    private_nh_.param("svg_mppi/gaussian_fitting_lambda", params.svg_mppi.gaussian_fitting_lambda, 0.1);
    private_nh_.param("svg_mppi/min_Vx_cov", params.svg_mppi.min_Vx_cov, 0.001);
    private_nh_.param("svg_mppi/max_Vx_cov", params.svg_mppi.max_Vx_cov, 0.1);
    // private_nh_.param("svg_mppi/min_Vy_cov", params.svg_mppi.min_Vy_cov, 0.001);
    // private_nh_.param("svg_mppi/max_Vy_cov", params.svg_mppi.max_Vy_cov, 0.1);
    // private_nh_.param("svg_mppi/min_Wz_cov", params.svg_mppi.min_Wz_cov, 0.001);
    // private_nh_.param("svg_mppi/max_Wz_cov", params.svg_mppi.max_Wz_cov, 0.1);

    // 강제로 svg_mppi 모드 사용
    mpc_solver_ptr_ = std::make_unique<mppi::cpu::SVGuidedMPPI>(params.common, params.svg_mppi);

    // twist 퍼블리셔만 설정 (환경 관련 퍼블리셔/서브스크라이버는 제거)
    pub_twist_cmd_ = nh_.advertise<geometry_msgs::Twist>(control_cmd_topic, 1);

    // 타이머 설정: control_sampling_time 주기로 timer_callback() 호출
    timer_control_ = nh_.createTimer(ros::Duration(control_sampling_time_), &MPPIControllerROS::timer_callback, this);
}

void MPPIControllerROS::timer_callback([[maybe_unused]] const ros::TimerEvent& te) {
    // 환경 데이터 없이 동작하도록 dummy initial state 사용
    mtx_.lock();
    stop_watch_.lap();
    mppi::cpu::State initial_state = mppi::cpu::State::Zero(); // dummy state (all zeros)
    const auto [updated_control_seq, collision_rate] = mpc_solver_ptr_->solve(initial_state);
    double calc_time = stop_watch_.lap();
    mtx_.unlock();

    // twist 메시지 생성: updated_control_seq의 첫 행 사용
    geometry_msgs::Twist twist_cmd;
    twist_cmd.linear.x  = updated_control_seq(0, CONTROL_SPACE::Vx);
    // // twist_cmd.linear.y  = updated_control_seq(0, CONTROL_SPACE::Vy);
    // // twist_cmd.angular.z = updated_control_seq(0, CONTROL_SPACE::Wz);
    pub_twist_cmd_.publish(twist_cmd);

    ROS_INFO_THROTTLE(2.0, "[MPPIControllerROS] Twist published: (Vx=%.2f)",
                      twist_cmd.linear.x);
}

}  // namespace mppi
