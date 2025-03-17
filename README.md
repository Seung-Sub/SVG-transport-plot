## 빌드와 동작은 기존 Holonomic-svg_mppi 레포와 동일하게 진행하면 됩니다.
1. summit_xl_sim을 동작
2. mppi_controller.launch를 동작
3. 일정 시간이 지난 후 ctrl+c로 동작 멈춤
4. 로그 확인은 script 폴더 내에 python 파일들로 plot 확인 가능

## cost함수 (optimal action distribution) 선택
stein_variational_guided_mppi.cpp에서 세 개의 func_calc_costs 함수 중 하나만 선택하여 활성화 후 나머지 주석처리
1. 가우시안 분포 (활성화 시 solve 위 optimal_mu와 sigma inv 등도 활성화)
2. 간단한 2차 함수
3. 간단한 4차 함수
각 함수 별 파라미터는 mppi_controller.yaml 내부 svg_mppi 항목안에 존재.
(optimal offset_step은 solve 마다 자동으로 평균 이동하도록 하는 값인데 현재는 디버깅을 위해 사용 안함)
