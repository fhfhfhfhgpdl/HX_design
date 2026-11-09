import numpy as np
import matplotlib.pyplot as plt

N_segments = 100

m_h = 3.5  
X_h = 60.0 
m_c = 3.0  
X_c = 57.0

D_vi = 0.0367
D_vo = 0.0377
k_wall = 401.0

Th_nodes = np.linspace(50.0, 80.0, N_segments + 1)
Tc_current = 45.0  


def calc_cp_libr(T, X):
    """LiBr 용액의 비열 계산 (단위: J/kg-K)"""
    x = X / 100.0
    term1 = -0.26642 + 10.97101 * x - 12.7661 * (x**2)
    term2 = 0.02502 - 0.08334 * x + 0.07541 * (x**2)
    return (term1 + term2 * T) * 1000

def get_h_tube(Th_avg):
    """
    Tube 측 열전달계수 계산 함수
    TODO: 엑셀에 구현했던 Re, Nu 식을 여기에 파이썬 수식으로 옮기세요.
    (아래는 플롯을 그리기 위한 임시 형태적 근사식입니다)
    """
    return 17000 + (Th_avg - 50) * 50 

def get_h_annular(Tc_avg):
    """
    Annular 측 열전달계수 계산 함수
    TODO: 엑셀에 구현했던 f_f, Nu 식을 여기에 파이썬 수식으로 옮기세요.
    """
    # 임시 근사 리턴
    return 13000 + (Tc_avg - 45) * 40

# ==========================================
# 3. 100구간 분할 수치해석 (Numerical Integration)
# ==========================================
h_tube_list = []
h_ann_list = []
U_list = []
L_total = 0.0

print("수치해석 계산 중...")

for i in range(N_segments):
    Th_current = Th_nodes[i]
    Th_next = Th_nodes[i+1]
    Th_avg = (Th_current + Th_next) / 2.0
    
    # 1) 해당 미소 구간의 물성치 및 방열량 계산
    Cp_h = calc_cp_libr(Th_avg, X_h)
    dQ = m_h * Cp_h * (Th_next - Th_current) # 해당 구간에서 넘겨준 열량 (W)
    
    # 2) 차가운 유체의 온도 변화 계산
    Cp_c = calc_cp_libr(Tc_current, X_c)
    dT_c = dQ / (m_c * Cp_c)
    Tc_next = Tc_current + dT_c
    Tc_avg = (Tc_current + Tc_next) / 2.0
    
    # 3) 실시간 열전달계수(h, U) 도출
    h_t = get_h_tube(Th_avg)
    h_a = get_h_annular(Tc_avg)