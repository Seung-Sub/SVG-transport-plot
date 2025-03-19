
# ===== 가이드 샘플 로그 시각화 (최적 제어 분포) =====
# !/usr/bin/env python3
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # ===================== 사용자 설정 =====================
# # CSV 파일 경로
# csv_guide_samples = "/tmp/svgd_guide_samples_log.csv"
# csv_best_particle = "/tmp/svgd_best_particle_log.csv"

# # 관심 대상: 특정 TimeStep 및 제어 입력 차원 선택 (예: TimeStep=0, Dim=0)
# time_step_target = 0
# dim_target = 0

# # Optimal Action Distribution (Gaussian) 파라미터 설정
# mu_opt = 0.0       # 평균
# sigma_opt = 1.0    # 표준편차
# lambda_value = 1.0 # lambda 값 (optimal distribution 계산에 사용)

# # Best Particle를 표시할 iteration 간격 (예: 30의 배수)
# best_iter_interval = 30

# # ===================== 최적 분포 (Gaussian) 계산 =====================
# x_vals = np.linspace(mu_opt - 4 * sigma_opt, mu_opt + 4 * sigma_opt, 300)
# optimal_pdf = (1 / (sigma_opt * np.sqrt(2 * np.pi))) * np.exp(-((x_vals - mu_opt) ** 2) / (2 * sigma_opt ** 2))

# def eval_optimal_pdf(x):
#     return (1 / (sigma_opt * np.sqrt(2 * np.pi))) * np.exp(-((x - mu_opt) ** 2) / (2 * sigma_opt ** 2))

# # ===================== CSV 파일 읽기 및 데이터 전처리 =====================
# df_samples = pd.read_csv(csv_guide_samples)
# df_samples = df_samples.sort_values("Iteration")
# # 가이드 샘플 로그: "Iteration", "SampleIdx", "TimeStep", "Dim", "Value", "Cost" 컬럼 존재
# df_filtered = df_samples[(df_samples["TimeStep"] == time_step_target) & (df_samples["Dim"] == dim_target)].copy()
# iterations = sorted(df_filtered["Iteration"].unique())

# # Best Particle 로그 읽기: "Iteration", "Row", "Col", "Value" 컬럼 존재
# df_best = pd.read_csv(csv_best_particle)

# # ===================== Figure 및 Axes 생성 =====================
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)

# # 초기 및 최종 iteration 데이터 (scatter로 표시)
# first_iter = iterations[0]
# last_iter = iterations[-1]
# df_first = df_filtered[df_filtered["Iteration"] == first_iter]
# df_last = df_filtered[df_filtered["Iteration"] == last_iter]

# ax.scatter(df_first["Value"].values, eval_optimal_pdf(df_first["Value"].values),
#            color="red", s=50, label=f"Iteration {first_iter}")
# ax.scatter(df_last["Value"].values, eval_optimal_pdf(df_last["Value"].values),
#            color="green", s=50, label=f"Iteration {last_iter}")

# # 현재 iteration의 가이드 샘플 scatter (색상 변경: "cyan")
# scatter_current = ax.scatter([], [], color="cyan", s=50, label="Current Iteration Samples")
# # Best Particle scatter (marker 크기 키움, marker "x")
# scatter_best = ax.scatter([], [], color="magenta", marker="x", s=120, label="Best Particle")

# ax.set_xlabel("Control Input Value")
# ax.set_ylabel("Probability Density")
# ax.set_title("SVGD Guide Sample Movement")
# ax.set_xlim(x_vals[0], x_vals[-1])
# ax.set_ylim(0, optimal_pdf.max() * 1.1)
# ax.grid(True)
# ax.legend(loc="upper right", fontsize=8)

# # ===================== 애니메이션 함수 =====================
# def init():
#     scatter_current.set_offsets(np.empty((0, 2)))
#     scatter_best.set_offsets(np.empty((0, 2)))
#     return scatter_current, scatter_best

# def update(frame):
#     it = iterations[frame]
#     df_it = df_filtered[df_filtered["Iteration"] == it]
#     x_points = df_it["Value"].values
#     y_points = eval_optimal_pdf(x_points)
#     scatter_current.set_offsets(np.column_stack((x_points, y_points)))
    
#     # Best Particle 표시: 현재 iteration이 best_iter_interval의 배수이면 표시
#     if it % best_iter_interval == 0:
#         df_bp_it = df_best[(df_best["Iteration"] == it) &
#                            (df_best["Row"] == time_step_target) &
#                            (df_best["Col"] == dim_target)]
#         if not df_bp_it.empty:
#             x_best = df_bp_it["Value"].values
#             y_best = eval_optimal_pdf(x_best)
#             scatter_best.set_offsets(np.column_stack((x_best, y_best)))
#         else:
#             scatter_best.set_offsets(np.empty((0, 2)))
#     else:
#         scatter_best.set_offsets(np.empty((0, 2)))
    
#     ax.set_title(f"SVGD Iteration: {it}")
#     return scatter_current, scatter_best

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=len(iterations),
#     init_func=init,
#     interval=300,
#     blit=False
# )

# # ===================== 플롯 이미지 저장 함수 =====================
# def save_plots():
#     # 초기 iteration 플롯 저장
#     init_iter = first_iter
#     df_init = df_filtered[df_filtered["Iteration"] == init_iter]
#     x_init = df_init["Value"].values
#     y_init = eval_optimal_pdf(x_init)
#     fig_init, ax_init = plt.subplots(figsize=(8, 5))
#     ax_init.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_init.scatter(x_init, y_init, color="red", s=50, label=f"Iteration {init_iter}")
#     ax_init.set_xlabel("Control Input Value")
#     ax_init.set_ylabel("Probability Density")
#     ax_init.set_title(f"Initial Iteration: {init_iter}")
#     ax_init.grid(True)
#     ax_init.legend(loc="upper right", fontsize=8)
#     fig_init.savefig("svgd_initial.png")
#     plt.close(fig_init)
    
#     # 마지막 iteration 플롯 저장
#     final_iter = last_iter
#     df_final = df_filtered[df_filtered["Iteration"] == final_iter]
#     x_final = df_final["Value"].values
#     y_final = eval_optimal_pdf(x_final)
#     fig_final, ax_final = plt.subplots(figsize=(8, 5))
#     ax_final.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_final.scatter(x_final, y_final, color="green", s=50, label=f"Iteration {final_iter}")
#     ax_final.set_xlabel("Control Input Value")
#     ax_final.set_ylabel("Probability Density")
#     ax_final.set_title(f"Final Iteration: {final_iter}")
#     ax_final.grid(True)
#     ax_final.legend(loc="upper right", fontsize=8)
#     fig_final.savefig("svgd_final.png")
#     plt.close(fig_final)
    
#     # Best Particle 플롯 저장 (사용자 선택 iteration)
#     try:
#         target_best_iter = int(input("Best Particle를 플롯할 iteration 번호를 입력하세요: "))
#     except:
#         target_best_iter = first_iter
#     df_bp = df_best[(df_best["Iteration"] == target_best_iter) &
#                     (df_best["Row"] == time_step_target) &
#                     (df_best["Col"] == dim_target)]
#     if df_bp.empty:
#         print(f"Iteration {target_best_iter}에 해당하는 best_particle 데이터가 없습니다. 초기 iteration 사용.")
#         df_bp = df_best[(df_best["Iteration"] == first_iter) &
#                         (df_best["Row"] == time_step_target) &
#                         (df_best["Col"] == dim_target)]
#         target_best_iter = first_iter
#     x_best = df_bp["Value"].values
#     y_best = eval_optimal_pdf(x_best)
#     fig_best, ax_best = plt.subplots(figsize=(8, 5))
#     ax_best.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_best.scatter(x_best, y_best, color="orange", s=70, label=f"Best Particle (Iteration {target_best_iter})")
#     ax_best.set_xlabel("Control Input Value")
#     ax_best.set_ylabel("Probability Density")
#     ax_best.set_title(f"Best Particle at Iteration {target_best_iter}")
#     ax_best.grid(True)
#     ax_best.legend(loc="upper right", fontsize=8)
#     fig_best.savefig(f"svgd_best_particle_iter_{target_best_iter}.png")
#     plt.close(fig_best)
    
#     # ----- Zoomed-in 플롯 저장 (x축 확대: mu_opt ± sigma_opt) -----
#     zoom_xlim = (mu_opt - sigma_opt, mu_opt + sigma_opt)
    
#     # Zoomed Initial
#     fig_init_zoom, ax_init_zoom = plt.subplots(figsize=(8, 5))
#     ax_init_zoom.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_init_zoom.scatter(x_init, y_init, color="red", s=50, label=f"Iteration {init_iter}")
#     ax_init_zoom.set_xlim(zoom_xlim)
#     ax_init_zoom.set_xlabel("Control Input Value")
#     ax_init_zoom.set_ylabel("Probability Density")
#     ax_init_zoom.set_title(f"Zoomed Initial Iteration: {init_iter}")
#     ax_init_zoom.grid(True)
#     ax_init_zoom.legend(loc="upper right", fontsize=8)
#     fig_init_zoom.savefig("svgd_initial_zoom.png")
#     plt.close(fig_init_zoom)
    
#     # Zoomed Final
#     fig_final_zoom, ax_final_zoom = plt.subplots(figsize=(8, 5))
#     ax_final_zoom.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_final_zoom.scatter(x_final, y_final, color="green", s=50, label=f"Iteration {final_iter}")
#     ax_final_zoom.set_xlim(zoom_xlim)
#     ax_final_zoom.set_xlabel("Control Input Value")
#     ax_final_zoom.set_ylabel("Probability Density")
#     ax_final_zoom.set_title(f"Zoomed Final Iteration: {final_iter}")
#     ax_final_zoom.grid(True)
#     ax_final_zoom.legend(loc="upper right", fontsize=8)
#     fig_final_zoom.savefig("svgd_final_zoom.png")
#     plt.close(fig_final_zoom)
    
#     # Zoomed Best Particle
#     fig_best_zoom, ax_best_zoom = plt.subplots(figsize=(8, 5))
#     ax_best_zoom.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
#     ax_best_zoom.scatter(x_best, y_best, color="orange", s=70, label=f"Best Particle (Iteration {target_best_iter})")
#     ax_best_zoom.set_xlim(zoom_xlim)
#     ax_best_zoom.set_xlabel("Control Input Value")
#     ax_best_zoom.set_ylabel("Probability Density")
#     ax_best_zoom.set_title(f"Zoomed Best Particle at Iteration {target_best_iter}")
#     ax_best_zoom.grid(True)
#     ax_best_zoom.legend(loc="upper right", fontsize=8)
#     fig_best_zoom.savefig(f"svgd_best_particle_iter_{target_best_iter}_zoom.png")
#     plt.close(fig_best_zoom)

# # ===================== 애니메이션 실행 및 플롯 저장 =====================
# plt.show()
# save_plots()

# 가우시안 분포 plot
#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===================== 사용자 설정 =====================
# CSV 파일 경로 (가이드 샘플 로그 및 best_particle 로그)
csv_guide_samples = "/tmp/svgd_guide_samples_log.csv"
csv_best_particle = "/tmp/svgd_best_particle_log.csv"

# 관심 대상: 특정 TimeStep 및 제어 입력 차원 선택 (예: TimeStep=0, Dim=0)
time_step_target = 0
dim_target = 0

# Optimal Action Distribution (Gaussian) 파라미터 설정
initial_mu_opt = 2.0       # 초기 평균
sigma_opt = 1.0            # 표준편차
optimal_offset_step = 0.0  # best_iter_interval마다 분포 평균이 이동하는 양

lambda_value = 1.0  # cost 계산에 사용되는 lambda 값 (분포 계산에도 사용)

# Best Particle 표시할 iteration 간격 (예: 30의 배수)
best_iter_interval = 30

# ===================== CSV 파일 읽기 및 데이터 전처리 =====================
# 가이드 샘플 로그 CSV (컬럼: Iteration, SampleIdx, TimeStep, Dim, Value, Cost 등)
df_samples = pd.read_csv(csv_guide_samples)
df_samples = df_samples.sort_values("Iteration")
df_filtered = df_samples[(df_samples["TimeStep"] == time_step_target) & (df_samples["Dim"] == dim_target)].copy()
df_filtered = df_filtered.sort_values("Iteration")
iterations = sorted(df_filtered["Iteration"].unique())

# best_particle 로그 CSV (컬럼: Iteration, Row, Col, Value)
df_best = pd.read_csv(csv_best_particle)
df_best = df_best.sort_values("Iteration")
# best_particle 데이터 중 관심 있는 time step (Row) 및 제어 차원 (Col) 필터
df_best_filtered = df_best[(df_best["Row"] == time_step_target) & (df_best["Col"] == dim_target)].copy()

# ===================== Optimal Distribution 함수 =====================
def compute_optimal_pdf(mu_opt, sigma_opt):
    # x 축: mu_opt ± 4*sigma_opt 범위로 설정
    x_vals = np.linspace(mu_opt - 4 * sigma_opt, mu_opt + 4 * sigma_opt, 300)
    pdf = (1 / (sigma_opt * np.sqrt(2 * np.pi))) * np.exp(-((x_vals - mu_opt) ** 2) / (2 * sigma_opt ** 2))
    return x_vals, pdf

def eval_optimal_pdf(x, mu_opt, sigma_opt):
    return (1 / (sigma_opt * np.sqrt(2 * np.pi))) * np.exp(-((x - mu_opt) ** 2) / (2 * sigma_opt ** 2))

# ===================== Figure 및 Axes 생성 =====================
# 초기 optimal distribution: 평균은 초기값으로 설정
# 여기서는 초기 분포의 평균을 initial_mu_opt로 사용합니다.
current_mu = initial_mu_opt  
x_vals, optimal_pdf = compute_optimal_pdf(current_mu, sigma_opt)

fig, ax = plt.subplots(figsize=(8, 5))
(line_optimal,) = ax.plot(x_vals, optimal_pdf, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)

# 초기 및 최종 iteration 데이터 (scatter 객체는 나중에 업데이트)
first_iter = iterations[0]
last_iter = iterations[-1]

# Scatter 객체 생성: 현재 iteration 가이드 샘플 (색상: cyan)와 best_particle (색상: magenta, marker: x)
scatter_current = ax.scatter([], [], color="cyan", s=50, label="Current Iteration Samples")
scatter_best = ax.scatter([], [], color="magenta", marker="x", s=120, label="Best Particle")

ax.set_xlabel("Control Input Value")
ax.set_ylabel("Probability Density")
ax.set_title("SVGD Guide Sample Movement")
ax.set_xlim(x_vals[0], x_vals[-1])
ax.set_ylim(0, optimal_pdf.max() * 1.1)
ax.grid(True)
ax.legend(loc="upper right", fontsize=8)

# ===================== 애니메이션 함수 =====================
def init():
    scatter_current.set_offsets(np.empty((0, 2)))
    scatter_best.set_offsets(np.empty((0, 2)))
    return scatter_current, scatter_best, line_optimal

def update(frame):
    it = iterations[frame]
    # 현재 iteration에 따른 optimal distribution의 평균을 best_iter_interval마다 업데이트:
    # 예를 들어, iteration이 0~29이면 평균은 initial_mu_opt, 30~59이면 initial_mu_opt + optimal_offset_step, etc.
    current_mu = initial_mu_opt + (it // best_iter_interval) * optimal_offset_step
    x_vals_dynamic, optimal_pdf_dynamic = compute_optimal_pdf(current_mu, sigma_opt)
    line_optimal.set_data(x_vals_dynamic, optimal_pdf_dynamic)
    
    # 현재 iteration의 가이드 샘플 데이터
    df_it = df_filtered[df_filtered["Iteration"] == it]
    x_points = df_it["Value"].values
    y_points = eval_optimal_pdf(x_points, current_mu, sigma_opt)
    scatter_current.set_offsets(np.column_stack((x_points, y_points)))
    
    # Best Particle 표시: 현재 iteration이 best_iter_interval의 배수이면 표시
    if it % best_iter_interval == 0:
        df_bp_it = df_best_filtered[df_best_filtered["Iteration"] == it]
        if not df_bp_it.empty:
            x_best = df_bp_it["Value"].values
            y_best = eval_optimal_pdf(x_best, current_mu, sigma_opt)
            scatter_best.set_offsets(np.column_stack((x_best, y_best)))
        else:
            scatter_best.set_offsets(np.empty((0, 2)))
    else:
        scatter_best.set_offsets(np.empty((0, 2)))
    
    ax.set_title(f"SVGD Iteration: {it}, Optimal μ: {current_mu:.2f}")
    return scatter_current, scatter_best, line_optimal

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(iterations),
    init_func=init,
    interval=300,
    blit=False
)

# ===================== 플롯 이미지 저장 함수 =====================
def save_plots():
    # 함수: 주어진 iteration에 대해 optimal distribution과 가이드 샘플 또는 best_particle 플롯 저장
    def save_plot(iter_val, df_data, marker_color, label_str, filename, zoom=False):
        current_mu = initial_mu_opt + (iter_val // best_iter_interval) * optimal_offset_step
        x_vals_local, optimal_pdf_local = compute_optimal_pdf(current_mu, sigma_opt)
        fig_local, ax_local = plt.subplots(figsize=(8, 5))
        ax_local.plot(x_vals_local, optimal_pdf_local, label="Optimal Distribution (Gaussian)", color="blue", linewidth=2)
        x_data = df_data["Value"].values
        y_data = eval_optimal_pdf(x_data, current_mu, sigma_opt)
        ax_local.scatter(x_data, y_data, color=marker_color, s=50, label=label_str)
        ax_local.set_xlabel("Control Input Value")
        ax_local.set_ylabel("Probability Density")
        ax_local.set_title(f"Iteration {iter_val}, Optimal μ: {current_mu:.2f}")
        ax_local.grid(True)
        ax_local.legend(loc="upper right", fontsize=8)
        if zoom:
            ax_local.set_xlim(current_mu - sigma_opt, current_mu + sigma_opt)
        fig_local.savefig(filename)
        plt.close(fig_local)
    
    # 초기 iteration 플롯 저장
    df_first = df_filtered[df_filtered["Iteration"] == first_iter]
    save_plot(first_iter, df_first, "red", f"Iteration {first_iter}", "svgd_initial.png")
    save_plot(first_iter, df_first, "red", f"Iteration {first_iter} (Zoomed)", "svgd_initial_zoom.png", zoom=True)
    
    # 최종 iteration 플롯 저장
    df_final = df_filtered[df_filtered["Iteration"] == last_iter]
    save_plot(last_iter, df_final, "green", f"Iteration {last_iter}", "svgd_final.png")
    save_plot(last_iter, df_final, "green", f"Iteration {last_iter} (Zoomed)", "svgd_final_zoom.png", zoom=True)
    
    # Best Particle 플롯 저장 (사용자가 입력한 iteration)
    try:
        target_best_iter = int(input("Best Particle를 플롯할 iteration 번호를 입력하세요: "))
    except:
        target_best_iter = first_iter
    df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]
    if df_bp.empty:
        print(f"Iteration {target_best_iter}에 해당하는 best_particle 데이터가 없습니다. 초기 iteration 사용.")
        target_best_iter = first_iter
        df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]
    save_plot(target_best_iter, df_bp, "orange", f"Best Particle (Iteration {target_best_iter})", 
              f"svgd_best_particle_iter_{target_best_iter}.png")
    save_plot(target_best_iter, df_bp, "orange", f"Best Particle (Iteration {target_best_iter}) (Zoomed)", 
              f"svgd_best_particle_iter_{target_best_iter}_zoom.png", zoom=True)

# ===================== 애니메이션 실행 및 플롯 저장 =====================
plt.show()
save_plots()

# 단순 2차 함수 plot
# #!/usr/bin/env python3
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # ===================== 사용자 설정 =====================
# # CSV 파일 경로 (가이드 샘플 로그 및 best_particle 로그)
# csv_guide_samples = "/tmp/svgd_guide_samples_log.csv"
# csv_best_particle = "/tmp/svgd_best_particle_log.csv"

# # 관심 대상: 특정 TimeStep 및 제어 입력 차원 선택 (예: TimeStep=0, Dim=0)
# time_step_target = 0
# dim_target = 0

# # ----- 새롭게 추가/수정할 포물선 파라미터 -----
# # y = -a * (x - mu)^2 + c  (단, y < 0이면 0으로 잘라낸다.)
# a_val = 0.25   # 포물선 폭/기울기
# c_val = 3.0   # 포물선 최대값 (x=mu에서의 peak)
# # --------------------------------------------

# # 기존 코드에 있던 변수 (mu_opt / sigma_opt 등)
# # 여기서는 mu_opt가 "peak 위치"로 사용될 것
# initial_mu_opt = 1.0       # 초기 "peak 위치"
# sigma_opt = 1.0            # (가우시안용 파라미터였으나, 여기서는 x축 범위 설정 등에만 활용)
# optimal_offset_step = 0.0  # best_iter_interval마다 peak가 이동하는 양

# lambda_value = 1.0  # cost 계산에 사용되는 lambda 값 (지금은 직접 사용 안 함)

# # Best Particle 표시할 iteration 간격 (예: 30의 배수)
# best_iter_interval = 30

# # ===================== CSV 파일 읽기 및 데이터 전처리 =====================
# df_samples = pd.read_csv(csv_guide_samples)
# df_samples = df_samples.sort_values("Iteration")
# df_filtered = df_samples[(df_samples["TimeStep"] == time_step_target) & 
#                          (df_samples["Dim"] == dim_target)].copy()
# df_filtered = df_filtered.sort_values("Iteration")
# iterations = sorted(df_filtered["Iteration"].unique())

# df_best = pd.read_csv(csv_best_particle)
# df_best = df_best.sort_values("Iteration")
# df_best_filtered = df_best[(df_best["Row"] == time_step_target) & 
#                            (df_best["Col"] == dim_target)].copy()

# # ===================== 새로 정의: 포물선 함수 =====================
# def compute_optimal_parabola(mu_opt, a, c):
#     """
#     x 범위: mu_opt ± 4*sigma_opt 로 설정 (원하는 범위로 조정 가능)
#     y = -a * (x - mu_opt)^2 + c  (음수가 되면 0으로 클리핑)
#     """
#     x_vals = np.linspace(mu_opt - 4 * sigma_opt, mu_opt + 4 * sigma_opt, 300)
#     y_vals = -a * (x_vals - mu_opt)**2 + c
#     y_vals = np.clip(y_vals, 0.0, None)  # 0 미만은 0으로 잘라냄
#     return x_vals, y_vals

# def eval_optimal_parabola(x, mu_opt, a, c):
#     """
#     주어진 x에 대해 포물선 값 계산
#     """
#     val = -a * (x - mu_opt)**2 + c
#     val = np.clip(val, 0.0, None)
#     return val

# # ===================== Figure 및 Axes 생성 =====================
# current_mu = initial_mu_opt  
# x_vals, parabola_vals = compute_optimal_parabola(current_mu, a_val, c_val)

# fig, ax = plt.subplots(figsize=(8, 5))
# (line_optimal,) = ax.plot(x_vals, parabola_vals,
#                           label="Optimal Parabola", color="blue", linewidth=2)

# first_iter = iterations[0]
# last_iter = iterations[-1]

# # Scatter 객체 생성
# scatter_current = ax.scatter([], [], color="cyan", s=50, label="Current Iteration Samples")
# scatter_best = ax.scatter([], [], color="magenta", marker="x", s=120, label="Best Particle")

# ax.set_xlabel("Control Input Value")
# ax.set_ylabel("Function Value")  # 예전엔 Probability Density였으나, 여기선 포물선 값
# ax.set_title("SVGD Guide Sample Movement (Parabola)")
# ax.set_xlim(x_vals[0], x_vals[-1])
# ax.set_ylim(0, parabola_vals.max() * 1.1)
# ax.grid(True)
# ax.legend(loc="upper right", fontsize=8)

# # ===================== 애니메이션 함수 =====================
# def init():
#     scatter_current.set_offsets(np.empty((0, 2)))
#     scatter_best.set_offsets(np.empty((0, 2)))
#     return scatter_current, scatter_best, line_optimal

# def update(frame):
#     it = iterations[frame]
#     # best_iter_interval마다 peak(=mu) 이동
#     current_mu = initial_mu_opt + (it // best_iter_interval) * optimal_offset_step
#     # 포물선 다시 계산
#     x_vals_dynamic, parabola_vals_dynamic = compute_optimal_parabola(current_mu, a_val, c_val)
#     line_optimal.set_data(x_vals_dynamic, parabola_vals_dynamic)

#     # 현재 iteration 가이드 샘플
#     df_it = df_filtered[df_filtered["Iteration"] == it]
#     x_points = df_it["Value"].values
#     # y값은 포물선 값
#     y_points = eval_optimal_parabola(x_points, current_mu, a_val, c_val)
#     scatter_current.set_offsets(np.column_stack((x_points, y_points)))

#     # Best Particle 표시 (iteration이 best_iter_interval 배수일 때)
#     if it % best_iter_interval == 0:
#         df_bp_it = df_best_filtered[df_best_filtered["Iteration"] == it]
#         if not df_bp_it.empty:
#             x_best = df_bp_it["Value"].values
#             y_best = eval_optimal_parabola(x_best, current_mu, a_val, c_val)
#             scatter_best.set_offsets(np.column_stack((x_best, y_best)))
#         else:
#             scatter_best.set_offsets(np.empty((0, 2)))
#     else:
#         scatter_best.set_offsets(np.empty((0, 2)))

#     ax.set_title(f"Iteration: {it}, Peak at μ={current_mu:.2f}")
#     return scatter_current, scatter_best, line_optimal

# ani = animation.FuncAnimation(
#     fig, update, frames=len(iterations),
#     init_func=init, interval=300, blit=False
# )

# # ===================== 플롯 이미지 저장 함수 =====================
# def save_plots():
#     def save_plot(iter_val, df_data, marker_color, label_str, filename, zoom=False):
#         current_mu = initial_mu_opt + (iter_val // best_iter_interval) * optimal_offset_step
#         x_vals_local, parabola_vals_local = compute_optimal_parabola(current_mu, a_val, c_val)

#         fig_local, ax_local = plt.subplots(figsize=(8, 5))
#         ax_local.plot(x_vals_local, parabola_vals_local, 
#                       label="Optimal Parabola", color="blue", linewidth=2)

#         x_data = df_data["Value"].values
#         y_data = eval_optimal_parabola(x_data, current_mu, a_val, c_val)
#         ax_local.scatter(x_data, y_data, color=marker_color, s=50, label=label_str)

#         ax_local.set_xlabel("Control Input Value")
#         ax_local.set_ylabel("Function Value")
#         ax_local.set_title(f"Iteration {iter_val}, Peak μ={current_mu:.2f}")
#         ax_local.grid(True)
#         ax_local.legend(loc="upper right", fontsize=8)

#         if zoom:
#             # peak 근처만 보고 싶다면 예: μ ± 1 범위로 제한
#             ax_local.set_xlim(current_mu - 1, current_mu + 1)

#         fig_local.savefig(filename)
#         plt.close(fig_local)

#     # 초기 iteration
#     df_first = df_filtered[df_filtered["Iteration"] == first_iter]
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter}", "svgd_initial.png")
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter} (Zoomed)", "svgd_initial_zoom.png", zoom=True)

#     # 최종 iteration
#     df_final = df_filtered[df_filtered["Iteration"] == last_iter]
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter}", "svgd_final.png")
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter} (Zoomed)", "svgd_final_zoom.png", zoom=True)

#     # Best Particle 플롯
#     try:
#         target_best_iter = int(input("Best Particle를 플롯할 iteration 번호를 입력하세요: "))
#     except:
#         target_best_iter = first_iter
#     df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]
#     if df_bp.empty:
#         print(f"Iteration {target_best_iter}에 해당하는 best_particle 데이터가 없습니다. 초기 iteration 사용.")
#         target_best_iter = first_iter
#         df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]

#     save_plot(target_best_iter, df_bp, "orange", 
#               f"Best Particle (Iteration {target_best_iter})", 
#               f"svgd_best_particle_iter_{target_best_iter}.png")
#     save_plot(target_best_iter, df_bp, "orange", 
#               f"Best Particle (Iteration {target_best_iter}) (Zoomed)", 
#               f"svgd_best_particle_iter_{target_best_iter}_zoom.png", zoom=True)

# # ===================== 애니메이션 실행 및 플롯 저장 =====================
# plt.show()
# save_plots()

# # 단순 4차 함수 plot
# #!/usr/bin/env python3
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # ===================== 사용자 설정 =====================
# # CSV 파일 경로 (가이드 샘플 로그 및 best_particle 로그)
# csv_guide_samples = "/tmp/svgd_guide_samples_log.csv"
# csv_best_particle = "/tmp/svgd_best_particle_log.csv"

# # 관심 대상: 특정 TimeStep 및 제어 입력 차원 선택 (예: TimeStep=0, Dim=0)
# time_step_target = 0
# dim_target = 0

# # ----- 4차 함수(멀티모달) 파라미터 -----
# # y = -a * ( ((x - mu)^2) - p^2 )^2 + c  (단, y < 0이면 0으로 잘라낸다.)
# # => x가 (mu ± p) 근처에서 봉우리(peak) 2개를 갖는 "W자" 모양
# a_val = 0.2   # 곡선의 폭/기울기
# p_val = 2   # ±p 지점에서 peak가 형성됨
# c_val = 4   # 최대 높이 (x가 mu ± p에서 y = c)
# # --------------------------------------

# # 기존 코드에서 사용하던 변수들
# initial_mu_opt = 1.0       # 중심 이동 (mu). (기존엔 1.0이었으나, 원하는 대로 조정 가능)
# sigma_opt = 3.0            # (가우시안 용도였지만 여기선 x 범위 설정에만 활용)
# optimal_offset_step = 0.0  # best_iter_interval마다 peak가 이동할 양

# # Best Particle 표시할 iteration 간격 (예: 30의 배수)
# best_iter_interval = 30

# # ===================== CSV 파일 읽기 및 데이터 전처리 =====================
# df_samples = pd.read_csv(csv_guide_samples)
# df_samples = df_samples.sort_values("Iteration")
# df_filtered = df_samples[
#     (df_samples["TimeStep"] == time_step_target) &
#     (df_samples["Dim"] == dim_target)
# ].copy()
# df_filtered = df_filtered.sort_values("Iteration")
# iterations = sorted(df_filtered["Iteration"].unique())

# df_best = pd.read_csv(csv_best_particle)
# df_best = df_best.sort_values("Iteration")
# df_best_filtered = df_best[
#     (df_best["Row"] == time_step_target) &
#     (df_best["Col"] == dim_target)
# ].copy()

# # ===================== 새로 정의: 4차(Quartic) 함수 =====================
# def compute_optimal_quartic(mu_opt, a, p, c):
#     """
#     x 범위: (mu_opt ± 4*sigma_opt) 구간을 300등분
#     y = -a * ( ((x - mu_opt)^2) - p^2 )^2 + c
#       (음수가 되면 0으로 잘라냄)
#     """
#     x_vals = np.linspace(mu_opt - 4 * sigma_opt, mu_opt + 4 * sigma_opt, 300)
#     y_vals = -a * ((x_vals - mu_opt)**2 - p**2)**2 + c
#     y_vals = np.clip(y_vals, 0.0, None)
#     return x_vals, y_vals

# def eval_optimal_quartic(x, mu_opt, a, p, c):
#     """
#     주어진 x에 대해 4차 함수 값 계산
#     """
#     val = -a * ((x - mu_opt)**2 - p**2)**2 + c
#     val = np.clip(val, 0.0, None)
#     return val

# # ===================== Figure 및 Axes 생성 =====================
# current_mu = initial_mu_opt
# x_vals, quartic_vals = compute_optimal_quartic(current_mu, a_val, p_val, c_val)

# fig, ax = plt.subplots(figsize=(8, 5))
# (line_optimal,) = ax.plot(
#     x_vals, quartic_vals,
#     label="Optimal Quartic (Multi-modal)",
#     color="blue", linewidth=2
# )

# first_iter = iterations[0]
# last_iter = iterations[-1]

# # Scatter 객체: 현재 iteration 샘플 & best_particle
# scatter_current = ax.scatter([], [], color="cyan", s=50, label="Current Iteration Samples")
# scatter_best = ax.scatter([], [], color="magenta", marker="x", s=120, label="Best Particle")

# ax.set_xlabel("Control Input Value")
# ax.set_ylabel("Function Value")
# ax.set_title("SVGD Guide Sample Movement (Quartic Multi-peak)")
# ax.set_xlim(x_vals[0], x_vals[-1])
# ax.set_ylim(0, quartic_vals.max() * 1.1)
# ax.grid(True)
# ax.legend(loc="upper right", fontsize=8)

# # ===================== 애니메이션 함수 =====================
# def init():
#     scatter_current.set_offsets(np.empty((0, 2)))
#     scatter_best.set_offsets(np.empty((0, 2)))
#     return scatter_current, scatter_best, line_optimal

# def update(frame):
#     it = iterations[frame]
#     # best_iter_interval마다 mu_opt가 이동 (optional)
#     current_mu = initial_mu_opt + (it // best_iter_interval) * optimal_offset_step
    
#     # 새로 4차 곡선 계산
#     x_vals_dyn, quartic_vals_dyn = compute_optimal_quartic(current_mu, a_val, p_val, c_val)
#     line_optimal.set_data(x_vals_dyn, quartic_vals_dyn)

#     # 현재 iteration 가이드 샘플
#     df_it = df_filtered[df_filtered["Iteration"] == it]
#     x_points = df_it["Value"].values
#     # y값 = quartic 함수
#     y_points = eval_optimal_quartic(x_points, current_mu, a_val, p_val, c_val)
#     scatter_current.set_offsets(np.column_stack((x_points, y_points)))

#     # Best Particle 표시
#     if it % best_iter_interval == 0:
#         df_bp_it = df_best_filtered[df_best_filtered["Iteration"] == it]
#         if not df_bp_it.empty:
#             x_best = df_bp_it["Value"].values
#             y_best = eval_optimal_quartic(x_best, current_mu, a_val, p_val, c_val)
#             scatter_best.set_offsets(np.column_stack((x_best, y_best)))
#         else:
#             scatter_best.set_offsets(np.empty((0, 2)))
#     else:
#         scatter_best.set_offsets(np.empty((0, 2)))

#     ax.set_title(f"Iteration: {it}, mu={current_mu:.2f}")
#     return scatter_current, scatter_best, line_optimal

# ani = animation.FuncAnimation(
#     fig, update,
#     frames=len(iterations),
#     init_func=init,
#     interval=300,
#     blit=False
# )

# # ===================== 플롯 이미지 저장 함수 =====================
# def save_plots():
#     def save_plot(iter_val, df_data, marker_color, label_str, filename, zoom=False):
#         current_mu = initial_mu_opt + (iter_val // best_iter_interval) * optimal_offset_step
#         x_vals_loc, quartic_vals_loc = compute_optimal_quartic(current_mu, a_val, p_val, c_val)

#         fig_local, ax_local = plt.subplots(figsize=(8, 5))
#         ax_local.plot(x_vals_loc, quartic_vals_loc,
#                       label="Optimal Quartic", color="blue", linewidth=2)

#         x_data = df_data["Value"].values
#         y_data = eval_optimal_quartic(x_data, current_mu, a_val, p_val, c_val)
#         ax_local.scatter(x_data, y_data, color=marker_color, s=50, label=label_str)

#         ax_local.set_xlabel("Control Input Value")
#         ax_local.set_ylabel("Function Value")
#         ax_local.set_title(f"Iteration {iter_val}, mu={current_mu:.2f}")
#         ax_local.grid(True)
#         ax_local.legend(loc="upper right", fontsize=8)

#         if zoom:
#             # peak 근처만 보고 싶다면, 예: mu ± (p+1) 범위
#             ax_local.set_xlim(current_mu - (p_val+1), current_mu + (p_val+1))

#         fig_local.savefig(filename)
#         plt.close(fig_local)

#     # 초기 iteration
#     df_first = df_filtered[df_filtered["Iteration"] == first_iter]
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter}", "svgd_initial.png")
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter} (Zoomed)", "svgd_initial_zoom.png", zoom=True)

#     # 최종 iteration
#     df_final = df_filtered[df_filtered["Iteration"] == last_iter]
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter}", "svgd_final.png")
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter} (Zoomed)", "svgd_final_zoom.png", zoom=True)

#     # Best Particle 플롯
#     try:
#         target_best_iter = int(input("Best Particle를 플롯할 iteration 번호를 입력하세요: "))
#     except:
#         target_best_iter = first_iter
#     df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]
#     if df_bp.empty:
#         print(f"Iteration {target_best_iter}에 해당하는 best_particle 데이터가 없습니다. 초기 iteration 사용.")
#         target_best_iter = first_iter
#         df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]

#     save_plot(target_best_iter, df_bp, "orange",
#               f"Best Particle (Iteration {target_best_iter})",
#               f"svgd_best_particle_iter_{target_best_iter}.png")
#     save_plot(target_best_iter, df_bp, "orange",
#               f"Best Particle (Iteration {target_best_iter}) (Zoomed)",
#               f"svgd_best_particle_iter_{target_best_iter}_zoom.png", zoom=True)

# # ===================== 애니메이션 실행 및 플롯 저장 =====================
# plt.show()
# save_plots()


# 가장 작은 cost를 가지는 서브 샘플 표시
#!/usr/bin/env python3
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # ===================== 사용자 설정 =====================
# # CSV 파일 경로 (가이드 샘플 로그 및 best_particle 로그)
# csv_guide_samples = "/tmp/svgd_guide_samples_log.csv"
# csv_best_particle = "/tmp/svgd_best_particle_log.csv"
# csv_best_subsample = "/tmp/best_subsample_log.csv"  # ★ 추가

# # 관심 대상: 특정 TimeStep 및 제어 입력 차원 선택 (예: TimeStep=0, Dim=0)
# time_step_target = 0
# dim_target = 0

# # ----- 4차 함수(멀티모달) 파라미터 -----
# a_val = 15.0   
# p_val = 1.3
# c_val = 43.0
# # --------------------------------------

# # 기존 코드에서 사용하던 변수들
# initial_mu_opt = 0.0
# sigma_opt = 3.0
# optimal_offset_step = 0.0

# # Best Particle 표시할 iteration 간격 (예: 30의 배수)
# best_iter_interval = 30

# # ===================== CSV 파일 읽기 및 데이터 전처리 =====================
# # (1) Guide samples
# df_samples = pd.read_csv(csv_guide_samples)
# df_samples = df_samples.sort_values("Iteration")
# df_filtered = df_samples[
#     (df_samples["TimeStep"] == time_step_target) &
#     (df_samples["Dim"] == dim_target)
# ].copy()
# df_filtered = df_filtered.sort_values("Iteration")
# iterations = sorted(df_filtered["Iteration"].unique())

# # (2) Best particle
# df_best = pd.read_csv(csv_best_particle)
# df_best = df_best.sort_values("Iteration")
# df_best_filtered = df_best[
#     (df_best["Row"] == time_step_target) &
#     (df_best["Col"] == dim_target)
# ].copy()

# # (3) ★ Best sub-sampler
# df_best_sub = pd.read_csv(csv_best_subsample)
# df_best_sub = df_best_sub.sort_values("Iteration")

# # time_step_target=0 인 행만 필터링 (만약 여러 time-step을 같이 보고 싶다면 로직 확장)
# df_best_sub_filtered = df_best_sub[df_best_sub["TimeStep"] == time_step_target].copy()

# # ===================== 새로 정의: 4차(Quartic) 함수 =====================
# def compute_optimal_quartic(mu_opt, a, p, c):
#     x_vals = np.linspace(mu_opt - 4*sigma_opt, mu_opt + 4*sigma_opt, 300)
#     y_vals = -a * ((x_vals - mu_opt)**2 - p**2)**2 + c
#     y_vals = np.clip(y_vals, 0.0, None)
#     return x_vals, y_vals

# def eval_optimal_quartic(x, mu_opt, a, p, c):
#     val = -a * ((x - mu_opt)**2 - p**2)**2 + c
#     val = np.clip(val, 0.0, None)
#     return val

# # ===================== Figure 및 Axes 생성 =====================
# current_mu = initial_mu_opt
# x_vals, quartic_vals = compute_optimal_quartic(current_mu, a_val, p_val, c_val)

# fig, ax = plt.subplots(figsize=(8, 5))
# (line_optimal,) = ax.plot(
#     x_vals, quartic_vals,
#     label="Optimal Quartic (Multi-modal)",
#     color="blue", linewidth=2
# )

# first_iter = iterations[0]
# last_iter = iterations[-1]

# # Scatter 객체 3가지:
# # 1) 현재 iteration 가이드 샘플 (색: cyan)
# # 2) Best Particle (색: magenta, marker='x')
# # 3) Best Sub-sampler (색: orange, marker='D')
# scatter_current = ax.scatter([], [], color="cyan", s=50, label="Current Iteration Samples")
# scatter_best = ax.scatter([], [], color="magenta", marker="x", s=120, label="Best Particle")
# scatter_best_sub = ax.scatter([], [], color="orange", marker="D", s=70, label="Best Sub-Sampler")

# ax.set_xlabel("Control Input Value")
# ax.set_ylabel("Function Value")
# ax.set_title("SVGD Guide Sample Movement (Quartic Multi-peak)")
# ax.set_xlim(x_vals[0], x_vals[-1])
# ax.set_ylim(0, quartic_vals.max() * 1.1)
# ax.grid(True)
# ax.legend(loc="upper right", fontsize=8)

# # ===================== 애니메이션 함수 =====================
# def init():
#     scatter_current.set_offsets(np.empty((0, 2)))
#     scatter_best.set_offsets(np.empty((0, 2)))
#     scatter_best_sub.set_offsets(np.empty((0, 2)))
#     return scatter_current, scatter_best, scatter_best_sub, line_optimal

# def update(frame):
#     it = iterations[frame]

#     # (1) mu 갱신 (optional)
#     current_mu = initial_mu_opt + (it // best_iter_interval) * optimal_offset_step
#     # 4차 곡선 재계산
#     x_vals_dyn, quartic_vals_dyn = compute_optimal_quartic(current_mu, a_val, p_val, c_val)
#     line_optimal.set_data(x_vals_dyn, quartic_vals_dyn)

#     # (2) 현재 iteration 가이드 샘플
#     df_it = df_filtered[df_filtered["Iteration"] == it]
#     x_points = df_it["Value"].values
#     y_points = eval_optimal_quartic(x_points, current_mu, a_val, p_val, c_val)
#     scatter_current.set_offsets(np.column_stack((x_points, y_points)))

#     # (3) Best Particle (조건부 표시)
#     if it % best_iter_interval == 0:
#         df_bp_it = df_best_filtered[df_best_filtered["Iteration"] == it]
#         if not df_bp_it.empty:
#             x_best = df_bp_it["Value"].values
#             y_best = eval_optimal_quartic(x_best, current_mu, a_val, p_val, c_val)
#             scatter_best.set_offsets(np.column_stack((x_best, y_best)))
#         else:
#             scatter_best.set_offsets(np.empty((0, 2)))
#     else:
#         scatter_best.set_offsets(np.empty((0, 2)))

#     # (4) ★ Best Sub-sampler 표시
#     # 동일 iteration에서 time_step=0의 행들 전부를 찍어본다.
#     df_sub_it = df_best_sub_filtered[df_best_sub_filtered["Iteration"] == it]
#     if not df_sub_it.empty:
#         x_sub = df_sub_it["ControlValue"].values
#         y_sub = eval_optimal_quartic(x_sub, current_mu, a_val, p_val, c_val)
#         scatter_best_sub.set_offsets(np.column_stack((x_sub, y_sub)))
#     else:
#         scatter_best_sub.set_offsets(np.empty((0, 2)))

#     ax.set_title(f"Iteration: {it}, mu={current_mu:.2f}")
#     return scatter_current, scatter_best, scatter_best_sub, line_optimal

# ani = animation.FuncAnimation(
#     fig, update,
#     frames=len(iterations),
#     init_func=init,
#     interval=300,
#     blit=False
# )

# # ===================== 플롯 이미지 저장 함수 =====================
# def save_plots():
#     def save_plot(iter_val, df_data, marker_color, label_str, filename, zoom=False):
#         current_mu = initial_mu_opt + (iter_val // best_iter_interval) * optimal_offset_step
#         x_vals_loc, quartic_vals_loc = compute_optimal_quartic(current_mu, a_val, p_val, c_val)

#         fig_local, ax_local = plt.subplots(figsize=(8, 5))
#         ax_local.plot(x_vals_loc, quartic_vals_loc,
#                       label="Optimal Quartic", color="blue", linewidth=2)

#         x_data = df_data["Value"].values
#         y_data = eval_optimal_quartic(x_data, current_mu, a_val, p_val, c_val)
#         ax_local.scatter(x_data, y_data, color=marker_color, s=50, label=label_str)

#         ax_local.set_xlabel("Control Input Value")
#         ax_local.set_ylabel("Function Value")
#         ax_local.set_title(f"Iteration {iter_val}, mu={current_mu:.2f}")
#         ax_local.grid(True)
#         ax_local.legend(loc="upper right", fontsize=8)

#         if zoom:
#             ax_local.set_xlim(current_mu - (p_val+1), current_mu + (p_val+1))

#         fig_local.savefig(filename)
#         plt.close(fig_local)

#     # 초기 iteration
#     df_first = df_filtered[df_filtered["Iteration"] == first_iter]
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter}", "svgd_initial.png")
#     save_plot(first_iter, df_first, "red", f"Iteration {first_iter} (Zoomed)", "svgd_initial_zoom.png", zoom=True)

#     # 최종 iteration
#     df_final = df_filtered[df_filtered["Iteration"] == last_iter]
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter}", "svgd_final.png")
#     save_plot(last_iter, df_final, "green", f"Iteration {last_iter} (Zoomed)", "svgd_final_zoom.png", zoom=True)

#     # Best Particle 플롯
#     try:
#         target_best_iter = int(input("Best Particle를 플롯할 iteration 번호를 입력하세요: "))
#     except:
#         target_best_iter = first_iter
#     df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]
#     if df_bp.empty:
#         print(f"Iteration {target_best_iter}에 해당하는 best_particle 데이터가 없습니다. 초기 iteration 사용.")
#         target_best_iter = first_iter
#         df_bp = df_best_filtered[df_best_filtered["Iteration"] == target_best_iter]

#     save_plot(target_best_iter, df_bp, "orange",
#               f"Best Particle (Iteration {target_best_iter})",
#               f"svgd_best_particle_iter_{target_best_iter}.png")
#     save_plot(target_best_iter, df_bp, "orange",
#               f"Best Particle (Iteration {target_best_iter}) (Zoomed)",
#               f"svgd_best_particle_iter_{target_best_iter}_zoom.png", zoom=True)

# # ===================== 애니메이션 실행 및 플롯 저장 =====================
# plt.show()
# save_plots()

