import math
import time

# Constantes orbitales reales del cometa Halley (NASA JPL)
E_HALLEY = 0.96714291
Q_HALLEY = 0.58597811
A_HALLEY = Q_HALLEY / (1.0 - E_HALLEY)
T_HALLEY = A_HALLEY ** 1.5


def tiempo_a_anomalia_media(t, T=T_HALLEY):
    return (2.0 * math.pi * t) / T


def f_kepler(E, e, M):
    return E - e * math.sin(E) - M


def df_kepler(E, e):
    return 1.0 - e * math.cos(E)


def posicion_orbital(E, a=A_HALLEY, e=E_HALLEY):
    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1.0 - e ** 2) * math.sin(E)
    r = a * (1.0 - e * math.cos(E))
    return x, y, r


def evaluar_metodo(nombre, func_metodo, args_metodo, e, M):
    """Mide tiempo de ejecución y estandariza la impresión exigida por el curso."""
    t0 = time.perf_counter()
    xk, erk, k, conv = func_metodo(*args_metodo)
    dt = time.perf_counter() - t0
    residuo = abs(f_kepler(xk, e, M))
    estado = "Convergió" if conv == 1 else "No convergió"
    print(f"{nombre:<15} | {estado:<12} | k={k:<4} | E*={xk:.10f} | |f(E*)|={residuo:.2e} | t={dt:.6f}s")


def experimento_halley_real(metodos):
    """Experimento 1: Órbita real de Halley en distintos instantes de tiempo."""
    tiempos = [1.0, 5.0, 15.0, T_HALLEY / 2.0]
    tol = 1e-10
    print("\n--- EXPERIMENTO 1: ÓRBITA REAL DEL COMETA HALLEY ---")

    for t in tiempos:
        M = tiempo_a_anomalia_media(t)
        f = lambda E: f_kepler(E, E_HALLEY, M)
        df = lambda E: df_kepler(E, E_HALLEY)
        print(f"\nt = {t:.2f} años | M = {M:.6f} rad | Intervalo: [{M - E_HALLEY:.4f}, {M + E_HALLEY:.4f}]")

        evaluar_metodo("Bisección", metodos.biseccion, (f, M - E_HALLEY, M + E_HALLEY, 200, tol), E_HALLEY, M)
        evaluar_metodo("Newton-Raphson", metodos.newton_raphson, (f, df, M, 100, tol), E_HALLEY, M)
        evaluar_metodo("Secante", metodos.secante, (f, M, M + E_HALLEY / 2.0, 100, tol), E_HALLEY, M)
        evaluar_metodo("Steffensen", metodos.steffensen, (f, M, 100, tol), E_HALLEY, M)


def experimento_caso_dificil(metodos):
    """Experimento 2: Excentricidad extrema (e -> 1) y M pequeña variando E0."""
    e_dif = 0.999
    M_peq = 0.01
    tol = 1e-10
    valores_E0 = [M_peq, 0.5, 1.5, math.pi]
    f = lambda E: f_kepler(E, e_dif, M_peq)
    df = lambda E: df_kepler(E, e_dif)

    print("\n--- EXPERIMENTO 2: CASO DIFÍCIL (e = 0.999, M = 0.01) ---")
    for E0 in valores_E0:
        print(f"\nValor inicial E0 = {E0:.4f}:")
        evaluar_metodo("Bisección", metodos.biseccion, (f, M_peq - e_dif, M_peq + e_dif, 200, tol), e_dif, M_peq)
        evaluar_metodo("Newton-Raphson", metodos.newton_raphson, (f, df, E0, 100, tol), e_dif, M_peq)
        evaluar_metodo("Secante", metodos.secante, (f, E0, E0 + e_dif / 2.0, 100, tol), e_dif, M_peq)
        evaluar_metodo("Steffensen", metodos.steffensen, (f, E0, 100, tol), e_dif, M_peq)


if __name__ == "__main__":
    # Cuando A 
    print(f"Constantes cargadas -> a = {A_HALLEY:.4f} UA, T = {T_HALLEY:.4f} años")