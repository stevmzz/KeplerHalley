"""Orquestador: ejecuta los experimentos del proyecto en orden."""
from src import metodos
from src.exp_convergencia import experimento_convergencia
from src.problema_halley import experimento_halley_real, experimento_caso_dificil
# from exp_barrido import experimento_barrido      # C
# from animacion import generar_animacion          # C


def correr_todo():
    """Corre los experimentos del proyecto."""
    experimento_convergencia()                  # A: métodos y orden de convergencia
    experimento_halley_real(metodos)            # B: Halley real
    experimento_caso_dificil(metodos)           # B: caso difícil
    # experimento_barrido()                     # C: barrido de excentricidad
    # generar_animacion()                       # C: animación de la órbita
    print("\nListo.")
