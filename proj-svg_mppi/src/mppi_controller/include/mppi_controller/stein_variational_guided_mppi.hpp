#pragma once
#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <functional> // [Modified] 사용자 정의 cost 함수용
#include <Eigen/Dense>
#include <array>
#include <grid_map_core/GridMap.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <memory>
#include <utility>

#include "mppi_controller/common.hpp"
#include "mppi_controller/mpc_base.hpp"
#include "mppi_controller/mpc_template.hpp"
#include "mppi_controller/prior_samples_with_costs.hpp"

namespace mppi {
namespace cpu {

/**
 * @brief Stein Variational Guided MPPI, K. Honda
 *
 * 이 클래스는 기존 MPC 기반의 동작 구조(노이즈 샘플링, gradient 근사, 가중평균 업데이트 등)는 유지하면서,
 * 각 샘플의 cost를 "optimal action distribution" (다변량 가우시안 분포)의 음의 로그우도(NLL)와,
 * 이전 제어 입력(prev_control_seq_)과의 차이를 반영하는 penalty 항을 결합하여 계산합니다.
 * 이 combined cost는 최종 가중치 계산 및 SVGD gradient 업데이트에 모두 사용됩니다.
 */
class SVGuidedMPPI : public MPCTemplate {
public:
    SVGuidedMPPI(const Params::Common& common_params, const Params::SVGuidedMPPI& svg_mppi_params);
    ~SVGuidedMPPI(){};

    /**
     * @brief solve mppi problem and return optimal control sequence
     * @param initial_state initial state
     * @return optimal control sequence and collision rate
     */
    std::pair<ControlSeq, double> solve(const State& initial_state) override;
    
    /**
     * @brief set obstacle map and reference map
     */
    void set_obstacle_map(const grid_map::GridMap& obstacle_map) override;
    // void set_obstacle_map(const nav_msgs::OccupancyGrid& obstacle_map) override;

    void set_reference_path(const nav_msgs::Path& reference_path) override;

    void set_single_goal(const geometry_msgs::PoseStamped& goal) override;

    /**
     * @brief get state sequence candidates and their weights, top num_samples
     * @return std::pair<std::vector<StateSeq>, std::vector<double>> state sequence candidates and their weights
     */
    std::pair<std::vector<StateSeq>, std::vector<double>> get_state_seq_candidates(const int& num_samples) const override;

    std::tuple<StateSeq, double, double, double> get_predictive_seq(const State& initial_state,
                                                                    const ControlSeq& control_input_seq) const override;

    ControlSeqCovMatrices get_cov_matrices() const override;

    ControlSeq get_control_seq() const override;

    std::pair<StateSeq, XYCovMatrices> get_proposed_state_distribution() const override;

    // --- [Modified] 사용자 정의 cost 함수 관련 ---
    /**
     * @brief 사용자 정의 cost 함수를 설정합니다.
     * @param func 사용자가 정의한 cost 함수. 입력: ControlSeq, 출력: cost (double)
     */
    void set_custom_cost_function(const std::function<double(const ControlSeq&)>& func);

    /**
     * @brief 사용자 정의 cost 함수를 사용하여 cost를 계산합니다.
     * @param sampler PriorSamplesWithCosts 객체 (가이드 샘플 집합)
     * @return 각 샘플에 대한 cost 값 벡터
     *
     * 기본적으로, cost는 각 샘플의 제어 시퀀스에 대해 다음과 같이 계산됩니다:
     * - 각 예측 시점에서 optimal action distribution (다변량 가우시안 분포, N(μ*, Σ*))에 대한 음의 로그우도(NLL)
     *   : 0.5 * (v_t - optimal_mu_)^T * inv_optimal_Sigma_ * (v_t - optimal_mu_)
     * - 이를 전체 시퀀스에 대해 평균한 값에, 이전 제어 입력(prev_control_seq_)과의 차이를 반영하는 penalty 항을 더한 결합 cost로 산출됩니다.
     */
    std::vector<double> custom_calc_sample_costs(const PriorSamplesWithCosts& sampler) const;

    std::vector<double> default_calc_sample_costs(const PriorSamplesWithCosts& sampler) const;

private:
    const size_t prediction_step_size_;  //!< prediction step size
    const int thread_num_;               //!< number of thread for parallel computation
    // MPPI parameters
    const double lambda_;  //!< temperature parameter [0, inf) of free energy.
    const double alpha_;
    const double non_biased_sampling_rate_;  //!< non-previous-control-input-biased sampling rate [0, 1]
    const double Vx_cov_;                 //!< covariance for control input dimension 1
    // const double Vy_cov_;                 //!< covariance for control input dimension 2
    // const double Wz_cov_;                 //!< covariance for control input dimension 3

    // Stein Variational MPC parameters
    const size_t sample_num_for_grad_estimation_;
    const double grad_lambda_;
    const double Vx_cov_for_grad_estimation_;
    // const double Vy_cov_for_grad_estimation_;
    // const double Wz_cov_for_grad_estimation_;
    const double svgd_step_size_;
    const int num_svgd_iteration_;
    const bool is_use_nominal_solution_;
    const bool is_covariance_adaptation_;
    const double gaussian_fitting_lambda_;
    const double min_Vx_cov_;
    const double max_Vx_cov_;
    // const double min_Vy_cov_;
    // const double max_Vy_cov_;
    // const double min_Wz_cov_;
    // const double max_Wz_cov_;
    // const std::string cost_function_type_;
    const double optimal_offset_step_;
    double a_;
    double b_;
    double c_;
    double p_;
    // int solve_count_;
    // Internal vars
    ControlSeq prev_control_seq_;
    ControlSeq nominal_control_seq_;
    std::vector<double> weights_ = {};  // for visualization
    Eigen::VectorXd optimal_mu_;
    Eigen::MatrixXd optimal_Sigma_;
    Eigen::MatrixXd inv_optimal_Sigma_;

    // Libraries
    std::unique_ptr<MPCBase> mpc_base_ptr_;
    std::unique_ptr<PriorSamplesWithCosts> prior_samples_ptr_;
    std::unique_ptr<PriorSamplesWithCosts> guide_samples_ptr_;
    std::vector<std::unique_ptr<PriorSamplesWithCosts>> grad_sampler_ptrs_;

    // --- [Modified] 사용자 정의 cost 함수 포인터 ---
    std::function<double(const ControlSeq&)> custom_cost_function_;
    std::function<double(const ControlSeq&)> default_quadratic_cost_;
    std::function<double(const ControlSeq&)> default_multimodal_cost_;
    std::function<double(const ControlSeq&)> default_optimal_cost_;

private:
    /**
     * @brief 기존 approx_grad_log_likelihood 함수:
     *        노이즈 샘플 재생성, cost 계산 (combined cost = optimal NLL + penalty 항), 
     *        exp(-cost/grad_lambda)로 가중치를 산출하여 각 시점별 gradient를 누적하는 구조를 유지합니다.
     */
    ControlSeq approx_grad_log_likelihood(const ControlSeq& mean_seq,
                                           const ControlSeq& noised_seq,
                                           const ControlSeqCovMatrices& inv_covs,
                                           const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs,
                                           PriorSamplesWithCosts* sampler,
                                           size_t sample_idx,
                                           int iteration_count) const;

    ControlSeqBatch approx_grad_posterior_batch(const PriorSamplesWithCosts& samples,
                                                const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs) const;

    std::pair<ControlSeq, ControlSeqCovMatrices> weighted_mean_and_sigma(const PriorSamplesWithCosts& samples,
                                                                         const std::vector<double>& weights) const;

    std::pair<ControlSeq, ControlSeqCovMatrices> estimate_mu_and_sigma(const PriorSamplesWithCosts& samples) const;

    std::vector<double> calc_weights(const PriorSamplesWithCosts& prior_samples_with_costs, const ControlSeq& nominal_control_seq) const;

    std::pair<double, double> gaussian_fitting(const std::vector<double>& x, const std::vector<double>& y) const;
};

}  // namespace cpu
}  // namespace mppi

