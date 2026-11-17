"""
Experimento de los métodos numéricos y su orden de convergencia.

Resuelve la ecuación de Kepler del cometa Halley (t = 7.5 años desde el
perihelio) con los cuatro métodos, imprime nombre, valores iniciales,
iteraciones, aproximación, error y tiempo, y estima el orden de
convergencia p de cada método a partir de su historial de aproximaciones.
"""
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metodos import biseccion, newton_raphson, secante, steffensen
from src.problema_halley import E_HALLEY, df_kepler, f_kepler, tiempo_a_anomalia_media

# --- Parámetros ---------------------------------------------------------
TOLERANCIA = 1e-10        # tolerancia de parada de todos los métodos
MAX_ITER = 100            # máximo de iteraciones (Newton, Secante, Steffensen)
MAX_ITER_BIS = 200        # máximo de iteraciones de la Bisección
REPETICIONES = 200        # repeticiones por método para medir el tiempo

CARPETA_RES = Path(__file__).resolve().parent.parent / "resultados"
METODOS = ["Bisección", "Newton-Raphson", "Secante", "Steffensen"]
TEORICO = {"Bisección": "1 (lineal, razón ~1/2)", "Newton-Raphson": "2",
           "Secante": "~1.618", "Steffensen": "2"}


def valores_iniciales(nombre, e, M):
    """
    Devuelve la tupla de valores iniciales del método `nombre`.

    La raíz de Kepler siempre está en [M - e, M + e], por eso ese es el
    intervalo de la Bisección. Los demás parten de E0 = M.
    """
    if nombre == "Bisección":
        return (M - e, M + e)
    if nombre == "Secante":
        return (M, M + e / 2.0)
    return (M,)               # Newton-Raphson y Steffensen


def llamar(nombre, f, fp, ini, historial=None):
    """Ejecuta el método `nombre` con valores iniciales `ini`; devuelve (xk, erk, k, conv)."""
    if nombre == "Bisección":
        return biseccion(f, ini[0], ini[1], MAX_ITER_BIS, TOLERANCIA, historial)
    if nombre == "Newton-Raphson":
        return newton_raphson(f, fp, ini[0], MAX_ITER, TOLERANCIA, historial)
    if nombre == "Secante":
        return secante(f, ini[0], ini[1], MAX_ITER, TOLERANCIA, historial)
    return steffensen(f, ini[0], MAX_ITER, TOLERANCIA, historial)


def mostrar_metodo(nombre, e, M):
    """Corre un método, mide su tiempo e imprime el resultado con el formato del enunciado."""
    f = lambda E: f_kepler(E, e, M)
    fp = lambda E: df_kepler(E, e)
    ini = valores_iniciales(nombre, e, M)

    # Se miden 5 lotes y se toma el mejor para reducir el ruido del sistema
    mejor = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(REPETICIONES):
            xk, erk, k, conv = llamar(nombre, f, fp, ini)
        mejor = min(mejor, time.perf_counter() - t0)

    print(f"  Método: {nombre}")
    print(f"    Valores iniciales : {', '.join(f'{v:.6f}' for v in ini)}")
    print(f"    Iteraciones       : {k}")
    print(f"    Aproximación E    : {xk:.12f}")
    print(f"    Error (criterio)  : {erk:.3e}")
    print(f"    Residuo |f(E)|    : {abs(f(xk)):.3e}")
    print(f"    Tiempo [s]        : {mejor / REPETICIONES:.3e}")
    print(f"    Convergió         : {'sí' if conv else 'no'}")


def raiz_referencia(e, M, tol=1e-14):
    """Raíz de referencia: Newton desde E0 = pi, iterado hasta precisión de máquina."""
    E = math.pi
    for _ in range(100):
        dE = f_kepler(E, e, M) / df_kepler(E, e)
        E -= dE
        if abs(dE) < tol:
            break
    return E


def orden_estimado(errs):
    """
    Estima el orden de convergencia con p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}).

    Parámetro: errs, lista de errores |E_k - E*|. Retorna la última p_k válida.
    """
    ps = []
    for i in range(1, len(errs) - 1):
        try:
            ps.append(math.log(errs[i + 1] / errs[i]) / math.log(errs[i] / errs[i - 1]))
        except (ValueError, ZeroDivisionError):
            pass
    return ps[-1] if ps else float("nan")


def experimento_convergencia():
    """Imprime resultados de los 4 métodos, estima p y guarda convergencia.png."""
    e = E_HALLEY
    M = tiempo_a_anomalia_media(7.5)
    f = lambda E: f_kepler(E, e, M)
    fp = lambda E: df_kepler(E, e)
    Eref = raiz_referencia(e, M)

    print("\n" + "=" * 72)
    print(f"MÉTODOS NUMÉRICOS: Halley, e = {e}, t = 7.5 años, M = {M:.6f}")
    print("=" * 72)
    for nombre in METODOS:
        mostrar_metodo(nombre, e, M)

    print("\n" + "=" * 72)
    print("ORDEN DE CONVERGENCIA ESTIMADO")
    print("=" * 72)
    print("p_k = ln(err_{k+1}/err_k) / ln(err_k/err_{k-1}),  err_k = |E_k - E*|")
    print("(solo se usan errores > 1e-12 para evitar ruido de redondeo)\n")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for nombre in METODOS:
        hist = []
        llamar(nombre, f, fp, valores_iniciales(nombre, e, M), hist)
        errs = [abs(x - Eref) for x in hist]
        util = [x for x in errs if x > 1e-12]
        if nombre == "Bisección":
            # El error de la bisección no baja monótono: se usa la razón media
            n = len(util)
            razon = (util[-1] / util[0]) ** (1.0 / (n - 1)) if n > 1 else float("nan")
            est = f"razón media de errores = {razon:.2f}"
        else:
            est = f"p estimado = {orden_estimado(util):.2f}"
        print(f"  {nombre:15s} teórico: {TEORICO[nombre]:22s} {est}")
        ax.semilogy(range(1, len(errs) + 1), np.maximum(errs, 1e-17), "o-", ms=3, label=nombre)

    ax.set_xlabel("iteración k")
    ax.set_ylabel("|E_k - E*|")
    ax.set_title("Error por iteración (Halley, t = 7.5 años)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    CARPETA_RES.mkdir(exist_ok=True)
    ruta = CARPETA_RES / "convergencia.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"\nFigura guardada: {ruta}")
