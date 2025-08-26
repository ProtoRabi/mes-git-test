
import numpy as np
import matplotlib.pyplot as plt

# 시뮬레이션 설정
nx, ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

dt = 0.1
T = 5.0
nt = int(T / dt)
mu = 0.1  # 점성 계수

# 초기 속도장
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# 에너지 기록
energy = []

# 시뮬레이션 루프
for n in range(nt):
    t = n * dt

    # 풍속 설정: 정지, 미풍, 정지, 강풍
    if t < 1.0 or (2.0 <= t < 3.0):
        wind_strength = 0.0
    elif 1.0 <= t < 2.0:
        wind_strength = 0.5
    else:
        wind_strength = 2.0

    omega = 1.0  # 회전 속도

    # 3대 선풍기: 서로 마주보는 구조
    u_wind = wind_strength * (
        np.sin(X - omega * t) +
        np.sin(Lx - X - omega * t) +
        np.sin(X - Lx/2 - omega * t)
    )
    v_wind = wind_strength * (
        np.sin(Y - omega * t) +
        np.sin(Ly - Y - omega * t) +
        np.sin(Y - Ly/2 - omega * t)
    )

    # 너울성 파도
    swell = 0.3 * np.sin(2 * X + 2 * Y - 0.5 * t)

    # 제채기 이벤트
    sneeze = np.zeros_like(u)
    if 1.5 <= t < 1.6:
        sneeze += 3.0 * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.1)

    # 속도장 업데이트 (단순화된 점성 확산 모델)
    u = u + dt * (mu * (np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0)) +
                mu * (np.roll(u, 1, axis=1) - 2*u + np.roll(u, -1, axis=1)))
    v = v + dt * (mu * (np.roll(v, 1, axis=0) - 2*v + np.roll(v, -1, axis=0)) +
                mu * (np.roll(v, 1, axis=1) - 2*v + np.roll(v, -1, axis=1)))

    # 외부 영향 추가
    u += u_wind + swell + sneeze
    v += v_wind + swell + sneeze

    # 난류 에너지 밀도 계산
    E = np.sum(u**2 + v**2)
    energy.append(E)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, nt), energy, label='Turbulent Energy')
plt.xlabel('Time')
plt.ylabel('Energy Density')
plt.title('Turbulent Energy over Time (3D Fan + Sneeze + Swell)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()