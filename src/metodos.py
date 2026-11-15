"""
Métodos numéricos para resolver f(x) = 0.

Todos devuelven la tupla (xk, erk, k, conv):
    xk   : aproximación final de la raíz
    erk  : error según el criterio de parada del método
    k    : número de iteraciones realizadas
    conv : 1 si se alcanzó la tolerancia, 0 si no

Criterios de parada:
    Newton-Raphson, Secante, Steffensen : |x_{k+1} - x_k| < tolerancia
    Bisección                           : |f(x_k)|        < tolerancia

Parámetro opcional `historial`: si se pasa una lista, se le agrega la
aproximación de cada iteración (sirve para estimar el orden de convergencia).
"""


def biseccion(f, a, b, maxiIteraciones, tolerancia, historial=None):
    """
    Bisección: reduce a la mitad el intervalo [a, b] que contiene la raíz.

    Parámetros:
        f               : función continua con f(a)*f(b) <= 0
        a, b            : extremos del intervalo inicial
        maxiIteraciones : máximo de iteraciones
        tolerancia      : se detiene cuando |f(xk)| < tolerancia
        historial       : lista opcional donde se guardan los xk

    Retorna (xk, erk, k, conv). Si no hay cambio de signo devuelve
    (None, None, None, 0).
    """
    a = float(a)
    b = float(b)
    fa = f(a)
    fb = f(b)

    # La raíz puede estar justo en un extremo
    if fa == 0:
        return a, 0.0, 0, 1
    if fb == 0:
        return b, 0.0, 0, 1

    # Sin cambio de signo no se garantiza la raíz
    if fa * fb > 0:
        return None, None, None, 0

    k = 0
    conv = 0
    xk = a
    erk = float("inf")

    while k < maxiIteraciones:
        k += 1
        xk = (a + b) / 2.0 # punto medio
        fxk = f(xk)
        erk = abs(fxk)
        if historial is not None:
            historial.append(xk)
        if erk < tolerancia:
            conv = 1
            break
        # Se conserva la mitad donde ocurre el cambio de signo
        if fa * fxk < 0:
            b = xk
        else:
            a = xk
            fa = fxk

    return xk, erk, k, conv


def newton_raphson(f, fp, x0, maxiIteraciones, tolerancia, historial=None):
    """
    Newton-Raphson: x_{k+1} = x_k - f(x_k) / f'(x_k).

    Parámetros:
        f, fp           : función y su derivada
        x0              : valor inicial
        maxiIteraciones : máximo de iteraciones
        tolerancia      : se detiene cuando |x_{k+1} - x_k| < tolerancia
        historial       : lista opcional donde se guardan los xk

    Retorna (xk, erk, k, conv).
    """
    k = 0
    conv = 0
    xk = float(x0)
    erk = float("inf")

    while k < maxiIteraciones:
        k += 1
        derivada = fp(xk)
        if derivada == 0: # evita dividir entre cero
            return xk, erk, k, conv

        xk_sig = xk - f(xk) / derivada
        erk = abs(xk_sig - xk)
        xk = xk_sig
        if historial is not None:
            historial.append(xk)

        if erk < tolerancia:
            conv = 1
            break

    return xk, erk, k, conv


def secante(f, x0, x1, maxiIteraciones, tolerancia, historial=None):
    """
    Secante: reemplaza f'(x_k) por la pendiente entre los dos últimos puntos.

    Parámetros:
        f               : función
        x0, x1          : dos valores iniciales distintos
        maxiIteraciones : máximo de iteraciones
        tolerancia      : se detiene cuando |x_{k+1} - x_k| < tolerancia
        historial       : lista opcional donde se guardan los xk

    Retorna (xk, erk, k, conv).
    """
    k = 0
    conv = 0
    x0 = float(x0)
    x1 = float(x1)
    erk = float("inf")

    while k < maxiIteraciones:
        k += 1
        f0 = f(x0)
        f1 = f(x1)
        denominador = f1 - f0
        if denominador == 0:
            # Si f(x1) ya es casi 0, la raíz ya se alcanzó
            if abs(f1) < tolerancia:
                return x1, 0.0, k, 1
            return x1, erk, k, conv

        x2 = x1 - f1 * (x1 - x0) / denominador
        erk = abs(x2 - x1)
        x0, x1 = x1, x2
        if historial is not None:
            historial.append(x1)

        if erk < tolerancia:
            conv = 1
            break

    return x1, erk, k, conv


def steffensen(f, x0, maxiIteraciones, tolerancia, historial=None):
    """
    Steffensen: Newton sin derivada, con f'(x_k) ~ (f(x_k + f(x_k)) - f(x_k)) / f(x_k).

    Parámetros:
        f               : función
        x0              : valor inicial
        maxiIteraciones : máximo de iteraciones
        tolerancia      : se detiene cuando |x_{k+1} - x_k| < tolerancia
        historial       : lista opcional donde se guardan los xk

    Retorna (xk, erk, k, conv).
    """
    k = 0
    conv = 0
    xk = float(x0)
    erk = float("inf")

    while k < maxiIteraciones:
        k += 1
        fxk = f(xk)
        denominador = f(xk + fxk) - fxk
        if denominador == 0:
            # Si f(xk) ya es casi 0, la raíz ya se alcanzó
            if abs(fxk) < tolerancia:
                return xk, 0.0, k, 1
            return xk, erk, k, conv

        xk_sig = xk - fxk ** 2 / denominador
        erk = abs(xk_sig - xk)
        xk = xk_sig
        if historial is not None:
            historial.append(xk)

        if erk < tolerancia:
            conv = 1
            break

    return xk, erk, k, conv
