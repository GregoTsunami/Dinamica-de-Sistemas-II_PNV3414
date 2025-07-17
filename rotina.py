from scipy.integrate import ode
from scipy.integrate import odeint
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd
from sympy.physics.mechanics import dynamicsymbols
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.fft import fft, fftfreq

#Parametros de entrada
params = {
    "M": 10000,    # kg   -> massa do corpo principal
    "mb1": 200,   # kg   -> massa da bóia lateral 1
    "mb2": 200,   # kg   -> massa da bóia lateral 2
    "mR1": 100,   # kg   -> massa do braço que liga mb1 ao corpo principal
    "mR2": 100,   # kg   -> massa do braço que liga mb2 ao corpo principal
    "L1": 3.0,    # m    -> comprimento do braço R1
    "L2": 3.0,    # m    -> comprimento do braço R2
    "k1": 15000,  # N/m  -> rigidez da mola associada à translação de x0
    "k2": 15000,  # N/m  -> rigidez da mola associada à translação de x0, segunda mola acoplada à x0
    "kT1": 3500,  # N·m/rad -> rigidez torcional do braço θ1
    "kT2": 3500,  # N·m/rad -> rigidez torcional do braço θ2
    "kT0": 10000, # N·m/rad -> rigidez torcional associada a φ (phi)
    "g": 9.81,    # m/s² -> aceleração gravitacional
    "A": 0.3,    # m    -> amplitude da onda incidente (altura da onda / 2)
    "f": 0.1,     # Hz   -> frequência da onda incidente
    "b": 1.0,     # m    -> braço de alavanca do momento restaurador (≈ metade do comprimento de x0)
}

# Parâmetros hidrodinâmicos para o cálculo da força do mar
params_hidrodinamicos = {
    "rho_agua": 1025,      # densidade da água (kg/m³)
    "g": 9.81,             # gravidade (m/s²)
    "T": 2.0,              # calado do corpo principal (m)
    "B": 2.0,              # boca do corpo principal (m)
    "L": 3.0               # comprimento do corpo principal (m)
}

#após alterar os valore dos parametros, calcular o novo valor de c via fórmula abaixo para que ele seja 5% do amortecimento crítico

#calculo de c
keq = params["k1"] + params["k2"]
m_total = params["M"] + params["mb1"] + params["mb2"] + params["mR1"] + params["mR1"]
params["c"] = 0.05 * 2 * np.sqrt(keq * m_total)
print(f"c = {params['c']:.2f}")


# Momento de inércia J do corpo principal (retângulo sólido em torno do centro)
params["J"] = 1/12 * params["M"]*(params_hidrodinamicos["L"]**2 + params_hidrodinamicos["B"]**2)
print(f"J = {params['J']:.2f}")

# Cálculo do coeficiente de amortecimento c_phi (5% do crítico)
zeta_phi = 0.05
params["c_phi"] = 2 * zeta_phi * np.sqrt(params["kT0"] * params["J"])
print(f"c_phi = {params['c_phi']:.2f}")

def sistema(t, y, p):
    x0, x0_dot = y[0], y[1]
    th1, th1_dot = y[2], y[3]
    th2, th2_dot = y[4], y[5]
    phi, phi_dot = y[6], y[7]

    # Cálculo da força do mar com modelo hidrodinâmico
    w_onda = 2 * np.pi * p["f"]
    k_onda = w_onda**2 / params_hidrodinamicos["g"]

    A_mar = (
        2
        * params_hidrodinamicos["rho_agua"]
        * params_hidrodinamicos["g"]
        * p["A"]
        * np.exp(-k_onda * params_hidrodinamicos["T"])
        * params_hidrodinamicos["B"]
        * (1 / k_onda)
        * np.sin(k_onda * params_hidrodinamicos["L"] / 2)
    )

    # Força F(t) modelada
    F_mar = A_mar * np.cos(w_onda * t)

    # # Cálculo da força de excitação (área molhada * pressão dinâmica)
    # area_molhada = params_hidrodinamicos["B"] * params_hidrodinamicos["L"]  # Boca x Comprimento
    # pressao_dinamica = 0.5 * params_hidrodinamicos["rho_agua"] * (2*np.pi*p["f"]*p["A"])**2
    # F_mar = area_molhada * pressao_dinamica * np.cos(2*np.pi*p["f"]*t)

    # Print da força do mar
    if t % 1.0 < 0.01:  # Print a cada ~1 segundo
        print(f"t = {t:.2f}s | F_mar = {F_mar:.2f} N")

    # Forças parciais para cada corpo
    F_M = F_mar            # total para M
    F_mb1 = 0.25 * F_mar   # 1/4 para mb1
    F_mb2 = 0.25 * F_mar   # 1/4 para mb2

    # Momentos restauradores com acoplamento via phi
    #M_res0 = -p["kT0"] * phi - p["c_phi"] * phi_dot
    M_res0 = (-p["kT0"] * phi   - (p["k1"] + p["k2"]) * p["b"] * phi - p["c_phi"] * phi_dot)
    M_res1 = -p["kT1"] * th1 - (p["k1"] + p["k2"]) * p["b"] * phi
    M_res2 = -p["kT2"] * th2 - (p["k1"] + p["k2"]) * p["b"] * phi


    # Cálculo das inércias
    I1 = p["mR1"]*(p["L1"]/2)**2 + (1/12)*p["mR1"]*p["L1"]**2 + p["mb1"]*p["L1"]**2 # -> inércia total
    I2 = p["mR2"]*(p["L2"]/2)**2 + (1/12)*p["mR2"]*p["L2"]**2 + p["mb2"]*p["L2"]**2

    # Matriz de massa
    M11 = p["M"] + p["mR1"] + p["mb1"] + p["mR2"] + p["mb2"] # -> coeficiente de \ddot{x}_0
    M12 = -(p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * np.sin(th1) # -> coeficiente de \ddot{th1}
    M13 = -(p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * np.sin(th2)  # -> coeficiente de \ddot{th2}

    M = np.array([
        [M11, M12, M13],
        [M12, I1, 0],
        [M13, 0, I2]
    ])

    # Vetor de forças
    f1 = (
        F_M
        + (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * th1_dot**2 * np.cos(th1)
        + (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * th2_dot**2 * np.cos(th2)
        - (p["k1"] + p["k2"]) * x0
        - p["c"] * x0_dot
    )

    f2 = (
        - (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * x0_dot * th1_dot * np.cos(th1)
        - (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * p["g"] * np.sin(th1)
        + M_res1
        + F_mb1 * p["L1"] * np.sin(th1)  # projeção da força na rotação
    )
    f3 = (
        - (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * x0_dot * th2_dot * np.cos(th2)
        - (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * p["g"] * np.sin(th2)
        + M_res2
        + F_mb2 * p["L2"] * np.sin(th2)  # projeção da força na rotação
    )
    f = np.array([f1, f2, f3])

    # Resolver sistema para as acelerações
    acc = np.linalg.solve(M, f)
    x0_ddot, th1_ddot, th2_ddot = acc

    # Equação para phi
    phi_ddot = (M_res0) / p["J"]

    # Prints de diagnóstico (posição, velocidade, ângulos)
    if t % 1.0 < 0.01:
        print(
            f"t = {t:.2f}s | "
            f"x0 = {x0:.3f} m | "
            f"x0_dot = {x0_dot:.3f} m/s | "
            f"θ1 = {np.degrees(th1):.2f}° | "
            f"θ2 = {np.degrees(th2):.2f}° | "
            f"phi = {np.degrees(phi):.2f}°"
        )

    return [x0_dot, x0_ddot, th1_dot, th1_ddot, th2_dot, th2_ddot, phi_dot, phi_ddot]

# Resolver com condições iniciais pequenas
t_eval = np.linspace(0, 120, 3000)
sol = solve_ivp(lambda t, y: sistema(t, y, params),
               (0, 120),
               [0, 0, 0.05, 0, 0.05, 0, 0.01, 0],  # θ1 e θ2 iniciam com 0.05 rad (~2.9°), repouso, φ = 0.11 rad (~5.7°)
               t_eval=t_eval,
               method='RK45')

#Pegar os dados da simulação para usar depois
# Tempo da simulação
t = sol.t  # [s]

# Variáveis do sistema (8 colunas)
x0      = sol.y[0]  # deslocamento
x0_dot  = sol.y[1]  # velocidade
th1     = sol.y[2]  # ângulo 1
th1_dot = sol.y[3]
th2     = sol.y[4]  # ângulo 2
th2_dot = sol.y[5]
phi     = sol.y[6]  # ângulo phi
phi_dot = sol.y[7]

# Plotar resultados
plt.figure(figsize=(14, 12))
plt.subplot(4, 2, 1); plt.plot(sol.t, sol.y[0]); plt.title("x0 [m]"); plt.grid()
plt.subplot(4, 2, 2); plt.plot(sol.t, sol.y[1]); plt.title("x0_dot [m/s]"); plt.grid()
plt.subplot(4, 2, 3); plt.plot(sol.t, sol.y[2]); plt.title("θ1 [rad]"); plt.grid()
plt.subplot(4, 2, 4); plt.plot(sol.t, sol.y[3]); plt.title("θ1_dot [rad/s]"); plt.grid()
plt.subplot(4, 2, 5); plt.plot(sol.t, sol.y[4]); plt.title("θ2 [rad]"); plt.grid()
plt.subplot(4, 2, 6); plt.plot(sol.t, sol.y[5]); plt.title("θ2_dot [rad/s]"); plt.grid()
plt.subplot(4, 2, 7); plt.plot(sol.t, sol.y[6]); plt.title("φ [rad]"); plt.grid()
plt.subplot(4, 2, 8); plt.plot(sol.t, sol.y[7]); plt.title("φ_dot [rad/s]"); plt.grid()
plt.tight_layout()
plt.show()

# Variáveis de estado
x0, x0_dot, x0_ddot = sp.symbols('x0 x0_dot x0_ddot')
th1, th1_dot, th1_ddot = sp.symbols('th1 th1_dot th1_ddot')
th2, th2_dot, th2_ddot = sp.symbols('th2 th2_dot th2_ddot')
phi, phi_dot, phi_ddot = sp.symbols('phi phi_dot phi_ddot')
Fmar = sp.symbols('Fmar')

# Parâmetros
M, mb1, mb2, mR1, mR2 = sp.symbols('M mb1 mb2 mR1 mR2')
L1, L2 = sp.symbols('L1 L2')
k1, k2 = sp.symbols('k1 k2')
kT1, kT2, kT0 = sp.symbols('kT1 kT2 kT0')
c, c_phi = sp.symbols('c c_phi')
g = sp.symbols('g')
b = sp.symbols('b')
J = sp.symbols('J')

# Inércias rotacionais
I1 = mR1*(L1/2)**2 + (1/12)*mR1*L1**2 + mb1*L1**2
I2 = mR2*(L2/2)**2 + (1/12)*mR2*L2**2 + mb2*L2**2

# Equações linearizadas

# (1) Translação do corpo principal
eq1 = (
    (M + mR1 + mb1 + mR2 + mb2) * x0_ddot
    - (mR1*L1/2 + mb1*L1) * th1_ddot
    - (mR2*L2/2 + mb2*L2) * th2_ddot
    - (k1 + k2)*x0
    - c * x0_dot
    + Fmar
)

# (2) Rotação do braço 1 (theta1)
eq2 = (
    I1 * th1_ddot
    - (mR1*L1/2 + mb1*L1) * x0_ddot
    + (mR1*L1/2 + mb1*L1) * g * th1
    + kT1 * th1
    + (k1 + k2)*b*phi
)

# (3) Rotação do braço 2 (theta2)
eq3 = (
    I2 * th2_ddot
    - (mR2*L2/2 + mb2*L2) * x0_ddot
    + (mR2*L2/2 + mb2*L2) * g * th2
    + kT2 * th2
    + (k1 + k2)*b*phi
)

# (4) Rotação do corpo principal (phi)
eq4 = (
    J * phi_ddot
    + c_phi * phi_dot
    + kT0 * phi
)

#Linearização utilizando séries de Taylor

# Variável de Laplace
s = sp.symbols('s')

# Inércias
I1 = params["mR1"]*(params["L1"]/2)**2 + (1/12)*params["mR1"]*params["L1"]**2 + params["mb1"]*params["L1"]**2
I2 = params["mR2"]*(params["L2"]/2)**2 + (1/12)*params["mR2"]*params["L2"]**2 + params["mb2"]*params["L2"]**2

#variável de amortecimento e braço de alavanca
b = params["b"]

# Matrizes com base nas equações linearizadas
M11 = params["M"] + params["mb1"] + params["mb2"] + params["mR1"] + params["mR2"]
M12 = -(params["mR1"]*params["L1"]/2 + params["mb1"]*params["L1"])
M13 = -(params["mR2"]*params["L2"]/2 + params["mb2"]*params["L2"])

# M̲: matriz de massa 4x4 (x0, th1, th2, phi)
M_lin = sp.Matrix([
    [M11, M12, M13, 0],
    [M12, I1, 0, 0],
    [M13, 0, I2, 0],
    [0,   0, 0, params["J"]]
])


# K̲: matriz de rigidez 4x4 com acoplamento via phi
k_sum = params["k1"] + params["k2"]
K_lin = sp.Matrix([
    [k_sum,   0,       0,      0],# x0
    [0,     params["kT1"], 0,   -k_sum * b],# θ1
    [0,     0,     params["kT2"], -k_sum * b],# θ2
    [0,   -k_sum * b, -k_sum * b, params["kT0"]]# φ
])


B = sp.Matrix([1, 0.25, 0.25, 0])
#B é a entrada F(t) = Fmar, aplicada em M,mb1 e mb2

# C̲: matriz de amortecimento (só em x0 e phi)
C_lin = sp.Matrix([
    [params["c"], 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, params["c_phi"]]
])

# Função de transferência (linearizada)
# G = ((s**2 * M_lin + K_lin).inv()) * B
#Inlcusao do amortecimento na FT

G_lin = ((s**2 * M_lin + s * C_lin + K_lin).inv()) * B
Gx0 = sp.simplify(G_lin[0])
Gth1 = sp.simplify(G_lin[1])
Gth2 = sp.simplify(G_lin[2])
Gphi = sp.simplify(G_lin[3])

# Frequência natural do sistema translacional aproximada
wn_x0 = float(sp.sqrt((params["k1"] + params["k2"]) / M11))

# Converter para TransferFunction (scipy)
def sympy_to_tf(expr):
    num, den = sp.fraction(expr)
    num_poly = sp.Poly(sp.expand(num), s)
    den_poly = sp.Poly(sp.expand(den), s)
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    return signal.TransferFunction(num_coeffs, den_coeffs)

# Converter Gx0
sys_x0 = sympy_to_tf(Gx0)

# Frequência adimensional: w/wn_x0
w = np.linspace(0.01, 3.0 * wn_x0, 1000)
_, mag, phase = signal.bode(sys_x0, w)

print("simplificação da expressao",sp.simplify(Gx0))
FT = sp.simplify(Gx0)

peaks, _ = signal.find_peaks(10**(mag / 20), prominence=0.0001)
print("Picos nas frequências normalizadas:", w[peaks] / wn_x0)

for i in peaks:
    print(f"Pico em ω/ωn ≈ {w[i]/wn_x0:.3f}, Amplitude ≈ {10**(mag[i]/20):.5f}")


# Plot em termos de w/wn
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(w / wn_x0, 10**(mag / 20), label='|Gx0(jω)|')
plt.ylabel('Amplitude')
plt.title('Função de Transferência Normalizada |X(ω)/F(ω)|')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(w / wn_x0, phase, label='∠Gx0(jω)')
plt.ylabel('Fase (graus)')
plt.xlabel('ω / ωₙ')
plt.grid(True)

plt.tight_layout()
plt.show()

# Converter as três funções de transferência
sys_x0 = sympy_to_tf(Gx0)
sys_th1 = sympy_to_tf(Gth1)
sys_th2 = sympy_to_tf(Gth2)

# Lista dos sistemas e seus nomes
sistemas = [
    (sys_x0, 'x₀', 'X₀(jω)/F(jω)'),
    (sys_th1, 'θ₁', 'Θ₁(jω)/F(jω)'),
    (sys_th2, 'θ₂', 'Θ₂(jω)/F(jω)')
]

# Frequência adimensional
w = np.linspace(0.01, 3.0 * wn_x0, 1000)
w_norm = w / wn_x0  # ω / ωₙ

# Loop para gerar gráfico e picos para cada modo
for sys, label, title in sistemas:
    _, mag, _ = signal.bode(sys, w)
    mag_lin = 10 ** (mag / 20)
    peaks, _ = signal.find_peaks(mag_lin, prominence=0.00001)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(w_norm, mag_lin, label=f'|{title}|')
    plt.plot(w_norm[peaks], mag_lin[peaks], 'rx', label='Picos')

    # Detalhar valores dos picos
    for i in peaks:
        freq_adim = w[i] / wn_x0
        amp = mag_lin[i]
        plt.annotate(f'{freq_adim:.2f}', (w_norm[i], amp),
                     textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title(f'Função de Transferência |{title}|')
    plt.xlabel('ω / ωₙ')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print picos no terminal
    print(f"\nPicos para {label}:")
    for i in peaks:
        print(f"  ω/ωₙ ≈ {w[i]/wn_x0:.3f}, Amplitude ≈ {mag_lin[i]:.5f}")

#RAO

s = sp.symbols('s')

def sympy_to_tf(expr):
    num, den = sp.fraction(expr)
    num_poly = sp.Poly(sp.expand(num), s)
    den_poly = sp.Poly(sp.expand(den), s)
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    return signal.TransferFunction(num_coeffs, den_coeffs)

# Convertendo cada função de transferência simbólica
sys_x0 = sympy_to_tf(Gx0)
sys_th1 = sympy_to_tf(Gth1)
sys_th2 = sympy_to_tf(Gth2)
sys_phi = sympy_to_tf(Gphi)

#definição das funções FT para cada variável
def FT_mod(sys, w):
    """Retorna |G(jw)| para uma dada função de transferência e vetor de frequência."""
    w = np.atleast_1d(w)
    _, mag, _ = signal.bode(sys, w)
    return 10**(mag / 20)

#cálculo dos RAOs

omega = np.linspace(0.1, 12.0, 1000)  # [rad/s]
A = params["A"]

# Cálculo dos RAOs padronizados (todos em escala 10^-3)
RAO_x0 = FT_mod(sys_x0, omega) / A * 1e3      # [mm/N]
RAO_th1 = FT_mod(sys_th1, omega) / A * 1e3    # [mrad/N]
RAO_th2 = FT_mod(sys_th2, omega) / A * 1e3    # [mrad/N]
RAO_phi = FT_mod(sys_phi, omega) / A * 1e3    # [mrad/N]

# Plot dos RAOs padronizados
plt.figure(figsize=(12, 5))
plt.plot(omega, RAO_x0, label='$x_0$ (mm/N)', linewidth=2)
plt.plot(omega, RAO_th1, label=r'$\theta_1$ (mrad/N)', linewidth=2)
plt.plot(omega, RAO_th2, label=r'$\theta_2$ (mrad/N)', linewidth=2)
plt.plot(omega, RAO_phi, label=r'$\phi$ (mrad/N)', linewidth=2)

plt.xlabel('Frequência (rad/s)', fontsize=12)
plt.ylabel('RAO (amplitude) (mm/N) ou (mrad/N)', fontsize=12)
plt.title('RAO dos Graus de Liberdade do Sistema', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


def imprimir_picos_padrao(rao, nome, unidade='mrad/N'):
    peaks, _ = signal.find_peaks(rao, prominence=0.1)  # Ajustado para escala µ
    print(f"\nPicos de {nome}:")
    for i in peaks:
        print(f"  ω = {omega[i]:.3f} rad/s | RAO = {rao[i]:.2f} {unidade}")

imprimir_picos_padrao(RAO_x0, 'x0', unidade='mm/N')
imprimir_picos_padrao(RAO_th1, 'θ1')
imprimir_picos_padrao(RAO_th2, 'θ2')
imprimir_picos_padrao(RAO_phi, 'ϕ')

# Análise espectral via FFT
#Parametros
dt = t[1] - t[0]
N = len(t)
fs = 1 / dt
f = fftfreq(N, dt)[:N // 2]
f_rad = 2 * np.pi * f  # Frequência em rad/s

def aplicar_fft(signal, nome, unidade, escala=1.0):
    """Aplica FFT ao sinal e plota o espectro."""
    y_fft = fft(signal - np.mean(signal))  # remove valor DC
    mag = 2.0 / N * np.abs(y_fft[0:N // 2])

    plt.figure(figsize=(10, 4))
    plt.plot(f_rad, mag * escala)
    plt.title(f'Espectro de {nome}')
    plt.xlabel('Frequência [rad/s]')
    plt.ylabel(f'Amplitude ({unidade})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Frequência dominante
    pico_idx = np.argmax(mag)
    freq_dominante = f[pico_idx]
    freq_rad_s = 2 * np.pi * freq_dominante

    print(f"Frequência dominante de {nome}: {freq_rad_s:.3f} rad/s")

# print(f"x0: tipo = {type(x0)}, shape = {np.shape(x0)}") # Remove this diagnostic print

#Aplicar FFT nos sinais
aplicar_fft(sol.y[0], 'x₀(t)', 'm')
aplicar_fft(sol.y[1], 'x₀_dot(t)', 'm/s')

aplicar_fft(sol.y[2], 'θ₁(t)', 'rad')
aplicar_fft(sol.y[3], 'θ₁_dot(t)', 'rad/s')

aplicar_fft(sol.y[4], 'θ₂(t)', 'rad')
aplicar_fft(sol.y[5], 'θ₂_dot(t)', 'rad/s')

aplicar_fft(sol.y[6], 'ϕ(t)', 'rad', escala=1e4)
aplicar_fft(sol.y[7], 'ϕ_dot(t)', 'rad/s')

#RAO via FFT

# Reconstruir F_mar(t)
omega_mar = 2 * np.pi * params["f"]
k_onda = omega_mar**2 / params["g"]
A_mar = (
    2 * params_hidrodinamicos["rho_agua"] * params["g"] * params["A"]
    * np.exp(-k_onda * params_hidrodinamicos["T"]) * params_hidrodinamicos["B"]
    * (1 / k_onda) * np.sin(k_onda * params_hidrodinamicos["L"] / 2)
)
F_mar = A_mar * np.cos(omega_mar * t)

# FFT parâmetros
dt = t[1] - t[0]
N = len(t)
fs = 1 / dt
f = fftfreq(N, dt)[:N // 2]
f_rad = 2 * np.pi * f

# Função para RAO via FFT
def rao_fft(entrada, saida, nome_saida, unidade_saida, escala_saida=1.0):
    fft_in = fft(entrada - np.mean(entrada))[:N // 2]
    fft_out = fft(saida - np.mean(saida))[:N // 2]
    mag_in = np.abs(fft_in)
    mag_out = np.abs(fft_out)

    rao = mag_out / mag_in

    plt.figure(figsize=(10, 4))
    plt.plot(f_rad, rao * escala_saida)
    plt.title(f'RAO via FFT: {nome_saida}(ω) / F_mar(ω)')
    plt.xlabel('Frequência [rad/s]')
    plt.ylabel(f'RAO ({unidade_saida}/N)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Frequência de pico da resposta
    pico_idx = np.argmax(rao)
    print(f"Pico de {nome_saida}: ω = {f_rad[pico_idx]:.3f} rad/s | RAO = {rao[pico_idx]*escala_saida:.3f} {unidade_saida}/N")

# Aplicar RAO para cada variável
rao_fft(F_mar, sol.y[0], 'x₀', 'm')
rao_fft(F_mar, sol.y[2], 'θ₁', 'rad')
rao_fft(F_mar, sol.y[4], 'θ₂', 'rad')
rao_fft(F_mar, sol.y[6], 'ϕ', 'rad')

# Massa do TMD (5% da massa do corpo principal)
params["m_tmd"] = 0.05 * params["M"]

# Frequência de sintonia (mesma da excitação)
omega_exc = 2 * np.pi * params["f"]
params["k_tmd"] = params["m_tmd"] * omega_exc**2

# Amortecimento do TMD (ótimo para minimizar vibração principal)
# Fórmula de Den Hartog para amortecimento:
# ξ_opt = √(3μ/(8(1+μ)^3)) onde μ = m_tmd / M
mu = params["m_tmd"] / params["M"]
params["c_tmd"] = 2 * np.sqrt(3 * mu / (8 * (1 + mu)**3)) * np.sqrt(params["m_tmd"] * params["k_tmd"])

print("\n=== PARÂMETROS DO TMD ===")
print(f"m_tmd = {params['m_tmd']:.2f} kg")
print(f"k_tmd = {params['k_tmd']:.2f} N/m")
print(f"c_tmd = {params['c_tmd']:.2f} N·s/m")
print(f"Frequência natural do TMD: {np.sqrt(params['k_tmd']/params['m_tmd'])/(2*np.pi):.4f} Hz")
print(f"Frequência de excitação: {params['f']:.4f} Hz")

def sistema_com_tmd(t, y, p):
    # Estados do sistema principal
    x0, x0_dot = y[0], y[1]
    th1, th1_dot = y[2], y[3]
    th2, th2_dot = y[4], y[5]
    phi, phi_dot = y[6], y[7]

    # Estados do TMD
    x_tmd, x_tmd_dot = y[8], y[9]

    # Cálculo da força do mar (mesmo que antes)
    w_onda = 2 * np.pi * p["f"]
    k_onda = w_onda**2 / params_hidrodinamicos["g"]

    A_mar = (
        2
        * params_hidrodinamicos["rho_agua"]
        * params_hidrodinamicos["g"]
        * p["A"]
        * np.exp(-k_onda * params_hidrodinamicos["T"])
        * params_hidrodinamicos["B"]
        * (1 / k_onda)
        * np.sin(k_onda * params_hidrodinamicos["L"] / 2)
    )

    F_mar = A_mar * np.cos(w_onda * t)

    # Forças parciais para cada corpo
    F_M = F_mar            # total para M
    F_mb1 = 0.25 * F_mar   # 1/4 para mb1
    F_mb2 = 0.25 * F_mar   # 1/4 para mb2

    # Momentos restauradores
    M_res0 = (-p["kT0"] * phi - (p["k1"] + p["k2"]) * p["b"] * phi - p["c_phi"] * phi_dot)
    M_res1 = -p["kT1"] * th1 - (p["k1"] + p["k2"]) * p["b"] * phi
    M_res2 = -p["kT2"] * th2 - (p["k1"] + p["k2"]) * p["b"] * phi

    # Força do TMD no corpo principal
    F_tmd = p["k_tmd"] * (x_tmd - x0) + p["c_tmd"] * (x_tmd_dot - x0_dot)

    # Cálculo das inércias
    I1 = p["mR1"]*(p["L1"]/2)**2 + (1/12)*p["mR1"]*p["L1"]**2 + p["mb1"]*p["L1"]**2
    I2 = p["mR2"]*(p["L2"]/2)**2 + (1/12)*p["mR2"]*p["L2"]**2 + p["mb2"]*p["L2"]**2

    # Matriz de massa (agora inclui o TMD)
    M11 = p["M"] + p["mR1"] + p["mb1"] + p["mR2"] + p["mb2"] + p["m_tmd"]  # massa total
    M12 = -(p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * np.sin(th1)
    M13 = -(p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * np.sin(th2)

    # Nota: O TMD não acopla diretamente com os ângulos, só com x0
    M = np.array([
        [M11, M12, M13],
        [M12, I1, 0],
        [M13, 0, I2]
    ])

    # Vetor de forças (atualizado com F_tmd)
    f1 = (
        F_M
        + (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * th1_dot**2 * np.cos(th1)
        + (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * th2_dot**2 * np.cos(th2)
        - (p["k1"] + p["k2"]) * x0
        - p["c"] * x0_dot
        - F_tmd  # Força do TMD atuando no corpo principal
    )

    f2 = (
        - (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * x0_dot * th1_dot * np.cos(th1)
        - (p["mR1"]*p["L1"]/2 + p["mb1"]*p["L1"]) * p["g"] * np.sin(th1)
        + M_res1
        + F_mb1 * p["L1"] * np.sin(th1)
    )

    f3 = (
        - (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * x0_dot * th2_dot * np.cos(th2)
        - (p["mR2"]*p["L2"]/2 + p["mb2"]*p["L2"]) * p["g"] * np.sin(th2)
        + M_res2
        + F_mb2 * p["L2"] * np.sin(th2)
    )

    f = np.array([f1, f2, f3])

    # Resolver sistema para as acelerações do sistema principal
    acc = np.linalg.solve(M, f)
    x0_ddot, th1_ddot, th2_ddot = acc

    # Equação para phi
    phi_ddot = M_res0 / p["J"]

    # Equação do TMD (dinâmica local)
    #####x_tmd_ddot = (F_tmd) / p["m_tmd"]
    x_tmd_ddot = (-p["k_tmd"] * (x_tmd - x0) - p["c_tmd"] * (x_tmd_dot - x0_dot)) / p["m_tmd"]

    # Diagnóstico (opcional)
    if t % 5.0 < 0.01:
        print(f"t = {t:.2f}s | x0 = {x0:.4f} m | x_tmd = {x_tmd:.4f} m | F_tmd = {F_tmd:.2f} N")

    return [x0_dot, x0_ddot,
            th1_dot, th1_ddot,
            th2_dot, th2_ddot,
            phi_dot, phi_ddot,
            x_tmd_dot, x_tmd_ddot]

t_eval = np.linspace(0, 120, 300) # Ajusta o tempo
y0 = [0, 0, 0.05, 0, 0.05, 0, 0.01, 0, 0, 0]  # + estados do TMD

sol = solve_ivp(lambda t, y: sistema_com_tmd(t, y, params),
               (0, 120),
               y0,
               t_eval=t_eval,
               method="Radau",   # Método implícito adequado
               atol=1e-8,
               rtol=1e-6)

# Solver mais simples para teste
# t_eval = np.linspace(0, 10, 1000)  # Tempo menor para teste
# y0 = [0, 0, 0.05, 0, 0.05, 0, 0.01, 0, 0, 0]

# sol = solve_ivp(
#     lambda t, y: sistema_com_tmd(t, y, params),
#     (0, 10),
#     y0,
#     t_eval=t_eval,
#     method='BDF',
#     rtol=1e-3,
#     atol=1e-6
# )

# Extrair resultados
t = sol.t
x0 = sol.y[0]
x_tmd = sol.y[8]

plt.figure(figsize=(14, 16))

# Sistema principal
plt.subplot(5, 2, 1)
plt.plot(t, x0)
plt.title("Deslocamento do Corpo Principal (x0)")
plt.ylabel("m")
plt.grid(True)

plt.subplot(5, 2, 2)
plt.plot(t, sol.y[1])
plt.title("Velocidade do Corpo Principal (dx0/dt)")
plt.ylabel("m/s")
plt.grid(True)

# TMD
plt.subplot(5, 2, 3)
plt.plot(t, x_tmd)
plt.title("Deslocamento do TMD (x_tmd)")
plt.ylabel("m")
plt.grid(True)

plt.subplot(5, 2, 4)
plt.plot(t, sol.y[9])
plt.title("Velocidade do TMD (dx_tmd/dt)")
plt.ylabel("m/s")
plt.grid(True)

# Ângulos
plt.subplot(5, 2, 5)
plt.plot(t, np.degrees(sol.y[2]))
plt.title("Ângulo θ1")
plt.ylabel("graus")
plt.grid(True)

plt.subplot(5, 2, 6)
plt.plot(t, np.degrees(sol.y[4]))
plt.title("Ângulo θ2")
plt.ylabel("graus")
plt.grid(True)

plt.subplot(5, 2, 7)
plt.plot(t, np.degrees(sol.y[6]))
plt.title("Ângulo φ")
plt.ylabel("graus")
plt.grid(True)

# Comparação x0 vs x_tmd
plt.subplot(5, 2, 8)
plt.plot(t, x0, 'b-', label='Corpo Principal')
plt.plot(t, x_tmd, 'r-', label='TMD')
plt.title("Comparação de Deslocamentos")
plt.ylabel("m")
plt.legend()
plt.grid(True)

# Força do TMD
F_tmd = params["k_tmd"] * (x_tmd - x0) + params["c_tmd"] * (sol.y[9] - sol.y[1])
plt.subplot(5, 2, 9)
plt.plot(t, F_tmd)
plt.title("Força do TMD no Corpo Principal")
plt.ylabel("N")
plt.xlabel("Tempo (s)")
plt.grid(True)

# Energia dissipada pelo TMD
potencia_tmd = F_tmd * (sol.y[9] - sol.y[1])
energia_dissipada = np.cumsum(potencia_tmd) * (t[1] - t[0])
plt.subplot(5, 2, 10)
plt.plot(t, energia_dissipada)
plt.title("Energia Dissipada pelo TMD")
plt.ylabel("J")
plt.xlabel("Tempo (s)")
plt.grid(True)

plt.tight_layout()
plt.show()

def simular_sem_tmd(params):
    t = np.linspace(0, 120, 3000)
    y0 = [0, 0,     # x0, x0_dot
          0.05, 0,  # th1, th1_dot
          0.05, 0,  # th2, th2_dot
          0.01, 0]  # phi, phi_dot

    sol = solve_ivp(lambda t, y: sistema(t, y, params), [0, 120], y0, t_eval=t, method='RK45')

    x0 = sol.y[0]   # deslocamento do corpo principal
    phi = sol.y[6]  # ângulo phi (opcional, caso queira usar)

    return t, x0, phi


# Simular SEM TMD
t_sem_tmd, x0_sem_tmd, _ = simular_sem_tmd(params)

# Simular COM TMD (usando sua função já existente)
t_com_tmd = np.linspace(0, 120, 3000)
y0_com_tmd = [0, 0, 0.05, 0, 0.05, 0, 0.01, 0, 0, 0]
sol_com_tmd = solve_ivp(lambda t, y: sistema_com_tmd(t, y, params), [0, 120], y0_com_tmd, t_eval=t_com_tmd)
x0_com_tmd = sol_com_tmd.y[0]
x_tmd = sol_com_tmd.y[8]

# Avaliar a amplitude (últimos 20% do sinal, ou seja, regime permanente)
N = len(t_sem_tmd)
janela = N // 5  # últimos 20%

amp_sem_tmd = np.max(np.abs(x0_sem_tmd[-janela:]))
amp_com_tmd = np.max(np.abs(x0_com_tmd[-janela:]))

print("\n=== DESEMPENHO DO TMD ===")
print(f"Amplitude máxima sem TMD: {amp_sem_tmd:.4f} m")
print(f"Amplitude máxima com TMD: {amp_com_tmd:.4f} m")
print(f"Redução de amplitude: {(1 - amp_com_tmd/amp_sem_tmd)*100:.2f}%")


# Amplitude máxima antes e depois do TMD
amp_sem_tmd = np.max(np.abs(x0[:500]))  # Primeiros 20% da simulação
amp_com_tmd = np.max(np.abs(x0[-500:])) # Últimos 20%


print("\n=== DESEMPENHO DO TMD ===")
print(f"Amplitude máxima inicial: {amp_sem_tmd:.4f} m")
print(f"Amplitude máxima final: {amp_com_tmd:.4f} m")
print(f"Redução: {(1 - amp_com_tmd/amp_sem_tmd)*100:.2f}%")

# FFT para análise espectral
def calc_fft(signal, t):
    N = len(t)
    T = t[1] - t[0]
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

# FFT para o corpo principal
freq, amp_x0 = calc_fft(x0, t)
freq_tmd, amp_tmd = calc_fft(x_tmd, t)

plt.figure(figsize=(12, 6))
plt.plot(freq, amp_x0, 'b-', label='Corpo Principal')
plt.plot(freq_tmd, amp_tmd, 'r-', label='TMD')
plt.axvline(x=params["f"], color='k', linestyle='--', label='Freq. Excitação')
plt.title("Espectro de Frequência")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim([0, 0.5])
plt.legend()
plt.grid(True)
plt.show()