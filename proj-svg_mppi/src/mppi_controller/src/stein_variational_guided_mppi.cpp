// stein_variational_guided_mppi.cpp
#include "mppi_controller/stein_variational_guided_mppi.hpp"
#include <ros/ros.h>
#include <fstream>

namespace mppi {
namespace cpu {

SVGuidedMPPI::SVGuidedMPPI(const Params::Common& common_params, const Params::SVGuidedMPPI& svg_mppi_params)
    : prediction_step_size_(static_cast<size_t>(common_params.prediction_step_size)),
      thread_num_(common_params.thread_num),
      lambda_(svg_mppi_params.lambda),
      alpha_(svg_mppi_params.alpha),
      non_biased_sampling_rate_(svg_mppi_params.non_biased_sampling_rate),
      Vx_cov_(svg_mppi_params.Vx_cov),
    //   Vy_cov_(svg_mppi_params.Vy_cov),
    //   Wz_cov_(svg_mppi_params.Wz_cov),
      sample_num_for_grad_estimation_(svg_mppi_params.sample_num_for_grad_estimation),
      grad_lambda_(svg_mppi_params.grad_lambda),
      Vx_cov_for_grad_estimation_(svg_mppi_params.Vx_cov_for_grad_estimation),
    //   Vy_cov_for_grad_estimation_(svg_mppi_params.Vy_cov_for_grad_estimation),
    //   Wz_cov_for_grad_estimation_(svg_mppi_params.Wz_cov_for_grad_estimation),
      svgd_step_size_(svg_mppi_params.svgd_step_size),
      num_svgd_iteration_(svg_mppi_params.num_svgd_iteration),
      is_use_nominal_solution_(svg_mppi_params.is_use_nominal_solution),
      is_covariance_adaptation_(svg_mppi_params.is_covariance_adaptation),
      gaussian_fitting_lambda_(svg_mppi_params.gaussian_fitting_lambda),
      min_Vx_cov_(svg_mppi_params.min_Vx_cov),
      max_Vx_cov_(svg_mppi_params.max_Vx_cov),
    //   min_Vy_cov_(svg_mppi_params.min_Vy_cov),
    //   max_Vy_cov_(svg_mppi_params.max_Vy_cov),
    //   min_Wz_cov_(svg_mppi_params.min_Wz_cov),
    //   max_Wz_cov_(svg_mppi_params.max_Wz_cov),
      optimal_offset_step_(svg_mppi_params.optimal_offset_step),
      a_(svg_mppi_params.a),
      b_(svg_mppi_params.b),
      c_(svg_mppi_params.c),
      p_(svg_mppi_params.p)
{
    const size_t sample_batch_num = static_cast<size_t>(svg_mppi_params.sample_batch_num);
    const size_t guide_sample_num = static_cast<size_t>(svg_mppi_params.guide_sample_num);
    const size_t sample_num_for_grad_estimation = static_cast<size_t>(svg_mppi_params.sample_num_for_grad_estimation);
    const size_t sample_num_for_cache = std::max(std::max(sample_batch_num, sample_num_for_grad_estimation), guide_sample_num);
    mpc_base_ptr_ = std::make_unique<MPCBase>(common_params, sample_num_for_cache);
    
    // 현재 컨트롤 입력 1차원 가정
    const double max_Vx = common_params.max_Vx;
    const double min_Vx = common_params.min_Vx;
    // const double max_Vy = common_params.max_Vy;
    // const double min_Vy = common_params.min_Vy;
    // const double max_Wz = common_params.max_Wz;
    // const double min_Wz = common_params.min_Wz;
    std::array<double, CONTROL_SPACE::dim> max_control_inputs = {max_Vx};
    std::array<double, CONTROL_SPACE::dim> min_control_inputs = {min_Vx};
    
    prior_samples_ptr_ = std::make_unique<PriorSamplesWithCosts>(sample_batch_num, prediction_step_size_, max_control_inputs, min_control_inputs,
                                                                   non_biased_sampling_rate_, thread_num_);
    guide_samples_ptr_ = std::make_unique<PriorSamplesWithCosts>(guide_sample_num, prediction_step_size_, max_control_inputs, min_control_inputs,
                                                                   non_biased_sampling_rate_, thread_num_);
    prev_control_seq_ = prior_samples_ptr_->get_zero_control_seq();
    nominal_control_seq_ = prior_samples_ptr_->get_zero_control_seq();
    
    const ControlSeqCovMatrices control_seq_cov_matrices = guide_samples_ptr_->get_constant_control_seq_cov_matrices({Vx_cov_});
    guide_samples_ptr_->random_sampling(guide_samples_ptr_->get_zero_control_seq(), control_seq_cov_matrices);
    
    for (size_t i = 0; i < sample_batch_num; i++) {
        grad_sampler_ptrs_.emplace_back(std::make_unique<PriorSamplesWithCosts>(sample_num_for_grad_estimation, prediction_step_size_,
                                                                                max_control_inputs, min_control_inputs, non_biased_sampling_rate_,
                                                                                thread_num_, i));
    }
    
    // optimal action distribution 선언 (나중에 수정할 수 있도록 해야함, 현재 0과 identity로 선언함 N(0, I))
    // optimal_mu_ = Eigen::VectorXd::Zero(CONTROL_SPACE::dim); // mean = 0
    optimal_mu_ = Eigen::VectorXd::Constant(CONTROL_SPACE::dim, 2.0); // mean 값 조정 부분
    optimal_Sigma_ = Eigen::MatrixXd::Identity(CONTROL_SPACE::dim, CONTROL_SPACE::dim);
    inv_optimal_Sigma_ = optimal_Sigma_.inverse();
}

std::pair<ControlSeq, double> SVGuidedMPPI::solve(const State& initial_state) {
    // --- [Modified] 로그 파일 추가: 가이드 샘플 상태 저장 (svgd_guide_samples_log.csv) ---
    static std::ofstream g_guide_samples_log;
    static bool g_is_guide_samples_log_open = false;
    if (!g_is_guide_samples_log_open) {
        g_guide_samples_log.open("/tmp/svgd_guide_samples_log.csv");
        g_guide_samples_log << "Iteration,SampleIdx,TimeStep,Dim,Value\n";
        g_is_guide_samples_log_open = true;
    }
    
    static std::ofstream g_cost_log_file;
    static bool g_is_cost_log_file_open = false;
    if (!g_is_cost_log_file_open) {
        g_cost_log_file.open("/tmp/svgd_cost_log.csv");
        g_cost_log_file << "Iteration,SampleIdx,Cost\n";
        g_is_cost_log_file_open = true;
    }
    static std::ofstream g_best_particle_log;
    static bool g_best_particle_log_opened = false;
    if (!g_best_particle_log_opened) {
        g_best_particle_log.open("/tmp/svgd_best_particle_log.csv");
        g_best_particle_log << "Iteration,Row,Col,Value\n";
        g_best_particle_log_opened = true;
    }
    static std::ofstream g_adaptive_cov_log;
    static bool g_adaptive_cov_log_opened = false;
    if (!g_adaptive_cov_log_opened) {
        g_adaptive_cov_log.open("/tmp/svgd_adaptive_cov_log.csv");
        g_adaptive_cov_log << "Iteration,TimeStep,Dim,CovValue\n";
        g_adaptive_cov_log_opened = true;
    }
    static int iter_count = 0;
    static int iteration_count = 0;

// 아래 3가지 func_calc_costs 함수 중 하나 선택하여 사용
// 가우시안 분포 사용
auto func_calc_costs = [this](const PriorSamplesWithCosts& sampler) -> std::vector<double> {
    std::vector<double> costs(sampler.get_num_samples(), 0.0);
    for (size_t i = 0; i < sampler.get_num_samples(); i++) {
        double total_nll = 0.0;
        for (size_t t = 0; t < prediction_step_size_ - 1; t++) {
            Eigen::VectorXd v_t = sampler.noised_control_seq_samples_[i].row(t).transpose();
            Eigen::VectorXd diff = v_t - optimal_mu_;
            double nll = 0.5 * diff.transpose() * inv_optimal_Sigma_ * diff;
            total_nll += nll;
        }
        double avg_nll = total_nll / static_cast<double>(prediction_step_size_ - 1);
        costs[i] = avg_nll;
    }
    return costs;
};

// 간단한 2차 함수 사용 (파라미터 mppi_controller.yaml 참고)
// auto func_calc_costs = [this](const PriorSamplesWithCosts& sampler) -> std::vector<double> 
// {
//     // 결과를 담을 벡터
//     std::vector<double> costs(sampler.get_num_samples(), 0.0);

//     // (주의) 여기선 1차원 제어라고 가정: v_t가 scalar
//     // 만약 dim>1이면, (v_t - b_)^2 대신 노름 ||v_t - b||^2 등을 사용해야 합니다.
    
//     for (size_t i = 0; i < sampler.get_num_samples(); i++) {
//         double total_cost = 0.0;

//         // 여러 time-step이 있다면, 각 t마다 cost 누적 (기존 구조를 따름)
//         for (size_t t = 0; t < prediction_step_size_ - 1; t++) {
//             // row(t)는 (1×dim) 행벡터, 여기선 dim=1이라고 가정
//             // transpose()하면 (dim×1) 열벡터
//             Eigen::VectorXd v_t = sampler.noised_control_seq_samples_[i].row(t).transpose();

//             // v_t[0] => 실제 스칼라
//             double x = v_t[0];  

//             // y = -a(x - b)^2 + c
//             // 1) 포물선 값이 음수가 되면 0으로 깎음( y >= 0 보장 )
//             double y = -this->a_ * (x - this->b_) * (x - this->b_) + this->c_;
//             if (y < 0.0) {
//                 y = 0.0;
//             }

//             // 2) y가 클수록 cost가 작게 만들고 싶음 -> 예: cost = -log(y + 작은 값)
//             //    y가 0이면 cost가 아주 커져서 penalty 작용
//             constexpr double EPS = 1e-8;
//             double cost_at_t = -std::log(y + EPS) + 0.5;

//             // time-step별 cost 누적
//             total_cost += cost_at_t;
//         }

//         // time-step 평균 (기존 코드처럼)
//         double avg_cost = total_cost / static_cast<double>(prediction_step_size_ - 1);
//         costs[i] = avg_cost;
//     }

//     return costs;
// };


// 간단한 4차 함수 사용 (파라미터 mppi_controller.yaml 참고)
// auto func_calc_costs = [this](const PriorSamplesWithCosts& sampler) -> std::vector<double> {
//     std::vector<double> costs(sampler.get_num_samples(), 0.0);

//     for (size_t i = 0; i < sampler.get_num_samples(); i++) {
//         double total_cost = 0.0;
//         // time-step마다 누적 (기존 code 구조와 동일)
//         for (size_t t = 0; t < prediction_step_size_ - 1; t++) {
//             // noised_control_seq_samples_[i].row(t)는 (1×dim) 행벡터
//             // 여기서는 dim=1 가정, v_t[0]가 실제 x 값
//             double x = sampler.noised_control_seq_samples_[i].row(t)[0];

//             // --- 4차 polynomial 값 계산 ---
//             //  y = -a_ * ( (x^2 - p_^2)^2 ) + c_
//             //  (만약 음수가 되면 0으로 클리핑)
//             double val = -a_ * std::pow(((x-b_) * (x-b_) - p_ * p_), 2) + c_;
//             if (val < 0.0) {
//                 val = 0.0; 
//             }

//             // --- y가 클수록 cost를 작게 ---
//             // 예시로 cost = -log(y + 아주 작은 양의값)
//             // y=0이면 cost 무한대로 커져서 penalty 효과
//             constexpr double EPS = 1e-8;
//             double cost_at_t = -std::log(val + EPS) + 0.5;

//             total_cost += cost_at_t;
//         }
//         // time-step 평균
//         costs[i] = total_cost / static_cast<double>(prediction_step_size_ - 1);
//     }
//     return costs;
// };



    // SVGD 루프
    std::vector<double> costs_history;
    std::vector<ControlSeq> control_seq_history;
    for (int i = 0; i < num_svgd_iteration_; i++) {
        const ControlSeqBatch grad_log_posterior = approx_grad_posterior_batch(*guide_samples_ptr_, func_calc_costs);
#pragma omp parallel for num_threads(thread_num_)
        for (size_t s = 0; s < guide_samples_ptr_->get_num_samples(); s++) {
            guide_samples_ptr_->noised_control_seq_samples_[s] += svgd_step_size_ * grad_log_posterior[s];
        }
        // [Modified] 각 iteration마다 가이드 샘플 상태를 CSV에 저장
        for (size_t sample_idx = 0; sample_idx < guide_samples_ptr_->get_num_samples(); sample_idx++) {
            const ControlSeq &cs = guide_samples_ptr_->noised_control_seq_samples_[sample_idx];
            for (int t = 0; t < cs.rows(); t++) {
                for (int d = 0; d < cs.cols(); d++) {
                    g_guide_samples_log << iteration_count << "," << sample_idx << "," << t << "," << d << "," << cs(t, d) << "\n";
                }
            }
        }

        const std::vector<double> costs = func_calc_costs(*guide_samples_ptr_);
        for (size_t s = 0; s < costs.size(); s++) {
            g_cost_log_file << iteration_count << "," << s << "," << costs[s] << "\n";
        }
        iteration_count++;

        costs_history.insert(costs_history.end(), costs.begin(), costs.end());
        control_seq_history.insert(control_seq_history.end(), guide_samples_ptr_->noised_control_seq_samples_.begin(),
                                    guide_samples_ptr_->noised_control_seq_samples_.end());
    }
    const std::vector<double> guide_costs = func_calc_costs(*guide_samples_ptr_);
    const size_t min_idx = std::distance(guide_costs.begin(), std::min_element(guide_costs.begin(), guide_costs.end()));
    const ControlSeq best_particle = guide_samples_ptr_->noised_control_seq_samples_[min_idx];

    // --- [Modified] best_particle 로그 저장 ---
    for (int row = 0; row < best_particle.rows(); row++) {
        for (int col = 0; col < best_particle.cols(); col++) {
            g_best_particle_log << iteration_count << "," << row << "," << col << "," << best_particle(row, col) << "\n";
        }
    }

    // Adaptive covariance 계산 (구조 유지)
    ControlSeqCovMatrices covs = prior_samples_ptr_->get_constant_control_seq_cov_matrices({Vx_cov_});
    if (is_covariance_adaptation_) {
        const std::vector<double> softmax_costs = softmax(costs_history, gaussian_fitting_lambda_, thread_num_);
        std::array<double, CONTROL_SPACE::dim> min_cov = {min_Vx_cov_};
        std::array<double, CONTROL_SPACE::dim> max_cov = {max_Vx_cov_};
        for (size_t i = 0; i < prediction_step_size_ - 1; i++) {
            covs[i] = Eigen::MatrixXd::Zero(CONTROL_SPACE::dim, CONTROL_SPACE::dim);
            for (size_t d = 0; d < CONTROL_SPACE::dim; d++) {
                std::vector<double> oneD_samples(control_seq_history.size());
                std::vector<double> q_star(control_seq_history.size());
                for (size_t idx = 0; idx < control_seq_history.size(); idx++) {
                    oneD_samples[idx] = control_seq_history[idx](i, d);
                    q_star[idx] = softmax_costs[idx];
                }
                const double sigma = gaussian_fitting(oneD_samples, q_star).second;
                const double sigma_clamped = std::clamp(sigma, min_cov[d], max_cov[d]);
                covs[i](d, d) = sigma_clamped;
            }
        }
    }
    iter_count++;

    for (size_t t = 0; t < (prediction_step_size_ - 1); t++) {
        for (size_t dim = 0; dim < CONTROL_SPACE::dim; dim++) {
            double cov_val = covs[t](dim, dim);
            g_adaptive_cov_log << iter_count << "," << t << "," << dim << "," << cov_val << "\n";
        }
    }

    // prior 샘플 업데이트: cost는 combined cost (optimal NLL + penalty)로 계산
    prior_samples_ptr_->random_sampling(prev_control_seq_, covs);
    const std::vector<double> prior_costs = func_calc_costs(*prior_samples_ptr_);
    prior_samples_ptr_->costs_ = prior_costs;
    if (is_use_nominal_solution_) {
        nominal_control_seq_ = best_particle;
    } else {
        nominal_control_seq_ = prior_samples_ptr_->get_zero_control_seq();
    }
    const std::vector<double> weights = calc_weights(*prior_samples_ptr_, nominal_control_seq_);
    weights_ = weights;  // 시각화를 위해 저장
    ControlSeq updated_control_seq = prior_samples_ptr_->get_zero_control_seq();
    for (size_t i = 0; i < prior_samples_ptr_->get_num_samples(); i++) {
        updated_control_seq += weights[i] * prior_samples_ptr_->noised_control_seq_samples_.at(i);
    }
    const double collision_rate = 0.0; // optimal 분포 기반이므로 collision 비용은 고려하지 않음.
    prev_control_seq_ = updated_control_seq;

    // solve마다 가우시안 평균값 조정(이동), 현재는 디버깅 중이니 사용 안함
    // optimal_mu_ += Eigen::VectorXd::Constant(CONTROL_SPACE::dim, optimal_offset_step_);
    // b_ += optimal_offset_step_;
    
    // 새로 샘플링하는 부분
    // for debug
    // {
    //     guide_samples_ptr_->random_sampling(nominal_control_seq_, covs);
    // }

    return std::make_pair(updated_control_seq, collision_rate);
}

// approx_grad_log_likelihood 함수: 기존 구조 그대로 유지하면서, 
// cost 계산 부분은 combined cost (optimal NLL + penalty)로 사용합니다.
ControlSeq SVGuidedMPPI::approx_grad_log_likelihood(const ControlSeq& mean_seq,
                                                      const ControlSeq& noised_seq,
                                                      const ControlSeqCovMatrices& inv_covs,
                                                      const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs,
                                                      PriorSamplesWithCosts* sampler,
                                                      size_t sample_idx,
                                                      int iteration_count) const {
    const ControlSeqCovMatrices grad_cov = sampler->get_constant_control_seq_cov_matrices({Vx_cov_for_grad_estimation_});
    sampler->random_sampling(noised_seq, grad_cov);
    sampler->costs_ = calc_costs(*sampler);
    std::vector<double> exp_costs(sampler->get_num_samples());
    ControlSeq sum_of_grads = mean_seq * 0.0;
    const ControlSeqCovMatrices sampler_inv_covs = sampler->get_inv_cov_matrices();

#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < sampler->get_num_samples(); i++) {
        double cost_with_control_term = sampler->costs_[i];
        
        // Control penalty 항 추가
        // for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
        //     const double diff_control_term = grad_lambda_ * (prev_control_seq_.row(j) - sampler->noised_control_seq_samples_[i].row(j))
        //                                      * inv_covs[j] *
        //                                      (prev_control_seq_.row(j) - sampler->noised_control_seq_samples_[i].row(j)).transpose();
        //     cost_with_control_term += diff_control_term;
        // }

        const double exp_cost = std::exp(-cost_with_control_term / grad_lambda_);
        exp_costs[i] = exp_cost;
        ControlSeq grad_log_gaussian = mean_seq * 0.0;
        for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
            grad_log_gaussian.row(j) = exp_cost * sampler_inv_covs[j] * (sampler->noised_control_seq_samples_[i] - noised_seq).row(j).transpose();
        }
        sum_of_grads += grad_log_gaussian;
    }
    const double sum_of_costs = std::accumulate(exp_costs.begin(), exp_costs.end(), 0.0);

    // === [디버그/로그용] 서브 샘플 중 cost 최소값 찾기 ===
    auto min_it = std::min_element(sampler->costs_.begin(), sampler->costs_.end());
    const size_t min_idx = std::distance(sampler->costs_.begin(), min_it);
    const double min_cost = *min_it;

    // (A) CSV 파일에 기록
    // "Iteration,GuideSampleIdx,BestSubSampleIdx,MinCost,TimeStep,ControlValue"
    static std::ofstream g_best_subsample_log;
    static bool g_is_best_subsample_log_open = false;
    if (!g_is_best_subsample_log_open) {
        g_best_subsample_log.open("/tmp/best_subsample_log.csv");
        g_best_subsample_log
            << "Iteration,GuideSampleIdx,BestSubSampleIdx,MinCost,TimeStep,ControlValue\n";
        g_is_best_subsample_log_open = true;
    }
    // (B) min_idx 서브 샘플의 time-step별 제어값을 여러 줄로 저장
    //     (제어 차원=1 가정)
    const ControlSeq& best_subsample_seq = sampler->noised_control_seq_samples_[min_idx];
    for (int t = 0; t < best_subsample_seq.rows(); t++) {
        double control_val = best_subsample_seq(t, 0);  // dim=1이므로 col=0
        g_best_subsample_log << iteration_count << ","
                             << sample_idx << ","
                             << min_idx << ","
                             << min_cost << ","
                             << t << ","
                             << control_val
                             << "\n";
    }
    // === [디버그/로그용] 끝 ===

    return sum_of_grads / (sum_of_costs + 1e-10);
}

ControlSeqBatch SVGuidedMPPI::approx_grad_posterior_batch(
    const PriorSamplesWithCosts& samples,
    const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs) const {
    
    // --- [Modified] 로그 파일 추가: 각 샘플의 gradient 저장 (svgd_gradient_log.csv) ---
    static std::ofstream g_gradient_log_file;
    static bool g_is_gradient_log_file_open = false;
    if (!g_is_gradient_log_file_open) {
        g_gradient_log_file.open("/tmp/svgd_gradient_log.csv"); 
        g_gradient_log_file << "Iteration,SampleIdx,Row,Col,GradValue\n";
        g_is_gradient_log_file_open = true;
    }
    static int iteration_count = 0;
    
    ControlSeqBatch grad_log_likelihoods = samples.get_zero_control_seq_batch();
    const ControlSeq mean = samples.get_mean();

    for (size_t i = 0; i < samples.get_num_samples(); i++) {
        const ControlSeq grad_log_likelihood = approx_grad_log_likelihood(
            mean, samples.noised_control_seq_samples_[i], samples.get_inv_cov_matrices(), calc_costs, grad_sampler_ptrs_.at(i).get(), i, iteration_count);
        grad_log_likelihoods[i] = grad_log_likelihood;
        
        // [Modified] 각 샘플의 gradient를 CSV에 저장
        for (size_t row = 0; row < grad_log_likelihood.rows(); row++) {
            for (size_t col = 0; col < grad_log_likelihood.cols(); col++) {
                double val = grad_log_likelihood(row, col);
                g_gradient_log_file << iteration_count << "," << i << "," << row << "," << col << "," << val << "\n";
            }
        }

    }
    iteration_count++;
    return grad_log_likelihoods;
}

std::pair<ControlSeq, ControlSeqCovMatrices> SVGuidedMPPI::weighted_mean_and_sigma(const PriorSamplesWithCosts& samples,
                                                                                   const std::vector<double>& weights) const {
    ControlSeq mean = samples.get_zero_control_seq();
    ControlSeqCovMatrices sigma = samples.get_zero_control_seq_cov_matrices();
    const ControlSeq prior_mean = samples.get_mean();
    const ControlSeqCovMatrices prior_inv_covs = samples.get_inv_cov_matrices();
#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < samples.get_num_samples(); i++) {
        mean += weights[i] * samples.noised_control_seq_samples_[i];
        const ControlSeq diff = samples.noised_control_seq_samples_[i] - prior_mean;
        for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
            sigma[j] += weights[i] * diff.row(j).transpose() * prior_inv_covs[j] * diff.row(j);
        }
    }
    return std::make_pair(mean, sigma);
}

std::pair<ControlSeq, ControlSeqCovMatrices> SVGuidedMPPI::estimate_mu_and_sigma(const PriorSamplesWithCosts& samples) const {
    ControlSeq mu = samples.get_zero_control_seq();
    ControlSeqCovMatrices sigma = samples.get_zero_control_seq_cov_matrices();
#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < samples.get_num_samples(); i++) {
        mu += samples.noised_control_seq_samples_[i];
    }
    mu /= static_cast<double>(samples.get_num_samples());
#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < samples.get_num_samples(); i++) {
        for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
            sigma[j] += (samples.noised_control_seq_samples_[i].row(j) - mu.row(j)).transpose() *
                        (samples.noised_control_seq_samples_[i].row(j) - mu.row(j));
        }
    }
    for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
        sigma[j] /= static_cast<double>(samples.get_num_samples());
        sigma[j] += 1e-5 * Eigen::MatrixXd::Identity(CONTROL_SPACE::dim, CONTROL_SPACE::dim);
    }
    return std::make_pair(mu, sigma);
}

std::vector<double> SVGuidedMPPI::calc_weights(const PriorSamplesWithCosts& prior_samples_with_costs,
                                               const ControlSeq& nominal_control_seq) const {
    const std::vector<double> costs_with_control_term =
        prior_samples_with_costs.get_costs_with_control_term(lambda_, alpha_, nominal_control_seq);
    return softmax(costs_with_control_term, lambda_, thread_num_);
}

std::pair<double, double> SVGuidedMPPI::gaussian_fitting(const std::vector<double>& x, const std::vector<double>& y) const {
    assert(x.size() == y.size());
    std::vector<double> y_hat(y.size(), 0.0);
    std::transform(y.begin(), y.end(), y_hat.begin(), [](double y) { return std::max(y, 1e-10); });
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < x.size(); i++) {
        const double y_hat_2 = y_hat[i] * y_hat[i];
        const double y_hat_log = std::log(y_hat[i]);
        A(0, 0) += y_hat_2;
        A(0, 1) += y_hat_2 * x[i];
        A(0, 2) += y_hat_2 * x[i] * x[i];
        A(1, 0) += y_hat_2 * x[i];
        A(1, 1) += y_hat_2 * x[i] * x[i];
        A(1, 2) += y_hat_2 * x[i] * x[i] * x[i];
        A(2, 0) += y_hat_2 * x[i] * x[i];
        A(2, 1) += y_hat_2 * x[i] * x[i] * x[i];
        A(2, 2) += y_hat_2 * x[i] * x[i] * x[i] * x[i];
        b(0) += y_hat_2 * y_hat_log;
        b(1) += y_hat_2 * x[i] * y_hat_log;
        b(2) += y_hat_2 * x[i] * x[i] * y_hat_log;
    }
    const Eigen::Vector3d u = A.colPivHouseholderQr().solve(b);
    const double eps = 1e-5;
    const double mean = -u(1) / (2.0 * std::min(u(2), -eps));
    const double variance = std::sqrt(1.0 / (2.0 * std::abs(u(2))));
    return std::make_pair(mean, variance);
}

// --- 다음 함수들은 MPCTemplate 인터페이스 준수를 위한 스텁 구현 ---

void SVGuidedMPPI::set_obstacle_map(const grid_map::GridMap& obstacle_map) {
    // optimal action distribution 기반에서는 사용하지 않음.
    // 필요하다면 빈 구현 또는 로그 출력.
}

void SVGuidedMPPI::set_reference_path(const nav_msgs::Path& reference_path) {
    // optimal action distribution 기반에서는 사용하지 않음.
}

void SVGuidedMPPI::set_single_goal(const geometry_msgs::PoseStamped& goal) {
    // optimal action distribution 기반에서는 사용하지 않음.
}

std::pair<std::vector<StateSeq>, std::vector<double>> SVGuidedMPPI::get_state_seq_candidates(const int& num_samples) const {
    // optimal action distribution 방식에서는 상태 시퀀스 후보를 생성하지 않으므로,
    // 빈 벡터를 반환합니다.
    return std::make_pair(std::vector<StateSeq>(), std::vector<double>());
}

std::tuple<StateSeq, double, double, double> SVGuidedMPPI::get_predictive_seq(const State& initial_state,
                                                                              const ControlSeq& control_input_seq) const {
    // optimal action distribution 방식에서는 예측 state 시퀀스를 계산하지 않으므로,
    // 기본 값(제로 상태 시퀀스 및 0 비용)을 반환합니다.
    StateSeq zero_state = StateSeq::Zero(prediction_step_size_, STATE_SPACE::dim);
    return std::make_tuple(zero_state, 0.0, 0.0, 0.0);
}

ControlSeqCovMatrices SVGuidedMPPI::get_cov_matrices() const {
    return prior_samples_ptr_->get_cov_matrices();
}

ControlSeq SVGuidedMPPI::get_control_seq() const {
    return nominal_control_seq_;
}

std::pair<StateSeq, XYCovMatrices> SVGuidedMPPI::get_proposed_state_distribution() const {
    // optimal action distribution 기반에서는 state distribution을 계산하지 않으므로,
    // 빈 값 또는 기본 값을 반환합니다.
    StateSeq zero_state = StateSeq::Zero(prediction_step_size_, STATE_SPACE::dim);
    XYCovMatrices xy_cov(prediction_step_size_, Eigen::MatrixXd::Zero(2, 2));
    return std::make_pair(zero_state, xy_cov);
}

}  // namespace cpu
}  // namespace mppi




