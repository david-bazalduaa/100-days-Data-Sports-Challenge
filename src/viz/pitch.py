"""
Day 13: Football Visualization Library

Objective:
Create pitch plotting functions (draw_pitch, plot_shots, plot_passes, plot_heatmap).
"""

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import numpy as np
import pandas as pd

# 1. DICCIONARIO DE TEMAS PREMIUM
THEMES = {
    'dark': {
        'fig_color': '#141414',
        'pitch_color': '#1e1e1e',
        'line_color': '#525252',
        'home_color': '#00d2ff',
        'away_color': '#ff0055',
        'text_color': '#ffffff'
    },
    'light': {
        'fig_color': '#f3f4f6',
        'pitch_color': '#ffffff',
        'line_color': '#cbd5e1',
        'home_color': '#0369a1',
        'away_color': '#be123c',
        'text_color': '#1e1e1e'
    }
}

def draw_pitch(pitch_type='statsbomb', orientation='horizontal', theme='dark', ax=None):
    """
    Configura y dibuja el lienzo del campo de fútbol.
    
    Args:
        pitch_type (str): Tipo de coordenadas ('statsbomb', 'opta', 'custom', etc.)
        orientation (str): 'horizontal' o 'vertical'
        theme (str): 'dark' o 'light'
        ax: (Opcional) Un axis de matplotlib si quieres dibujarlo en un subplot.
        
    Returns:
        fig, ax, pitch: Objetos de matplotlib y mplsoccer para seguir interactuando.
    """
    if theme not in THEMES:
        raise ValueError(f"Theme '{theme}' no está soportado. Usa 'dark' o 'light'.")
        
    t = THEMES[theme]
    
    # Seleccionamos la clase de mplsoccer adecuada según la orientación
    PitchClass = VerticalPitch if orientation == 'vertical' else Pitch
    
    pitch = PitchClass(
        pitch_type=pitch_type,
        pitch_color=t['pitch_color'],
        line_color=t['line_color'],
        linewidth=1.5,
        half=False # Útil cambiar a True para mapas de tiros
    )
    
    # Si no nos pasan un eje, creamos una figura nueva
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.patch.set_facecolor(t['fig_color'])
    else:
        # Si nos pasan un eje (ej. un subplot), dibujamos sobre ese eje
        pitch.draw(ax=ax)
        fig = ax.figure
        fig.patch.set_facecolor(t['fig_color'])
    
    return fig, ax, pitch

def plot_shots(shots_df, ax, pitch_obj, theme='dark'):
    """
    Dibuja los tiros en el campo. El tamaño del punto se escala según el xG.
    
    Args:
        shots_df (pd.DataFrame): DataFrame con ['x', 'y', 'xg', 'is_goal']
        ax: Matplotlib axis retornado por draw_pitch.
        pitch_obj: Objeto Pitch de mplsoccer retornado por draw_pitch.
        theme: 'dark' o 'light'.
    """
    if shots_df.empty:
        return
        
    t = THEMES[theme]
    
    # Extraemos coordenadas y simulamos variables si no existen (para testing robusto)
    x = shots_df.get('x', pd.Series(dtype=float))
    y = shots_df.get('y', pd.Series(dtype=float))
    
    # ESCALADO DE xG: Multiplicamos por 500 y sumamos 50 para que el área del 
    # círculo sea proporcional a la probabilidad de gol y los tiros de bajo xG sean visibles.
    xg = shots_df.get('xg', pd.Series([0.1]*len(shots_df))) 
    sizes = xg * 500 + 50  
    
    goals = shots_df.get('is_goal', pd.Series([False]*len(shots_df)))
    
    # 1. Dibujar Goles (Rellenos)
    pitch_obj.scatter(
        x[goals], y[goals],
        s=sizes[goals],
        color=t['home_color'],
        edgecolors=t['text_color'], # Borde de alto contraste
        linewidth=1.5,
        alpha=0.9,
        ax=ax,
        label='Gol'
    )
    
    # 2. Dibujar Tiros Fallados (Huecos)
    pitch_obj.scatter(
        x[~goals], y[~goals],
        s=sizes[~goals],
        color=t['pitch_color'],     # Transparente (color del campo)
        edgecolors=t['home_color'], # Solo el contorno tiene color
        linewidth=1.5,
        alpha=0.6,
        ax=ax,
        label='No Gol'
    )
    
    # Añadimos leyenda estéticamente adaptada al tema
    ax.legend(facecolor=t['fig_color'], edgecolor='none', labelcolor=t['text_color'], loc='upper left')

def plot_passes(passes_df, ax, pitch_obj, theme='dark'):
    """
    Dibuja vectores de pases (flechas).
    Aplica técnicas visuales (transparencia) para evitar 'Spaghetti Plots'.
    
    Args:
        passes_df (pd.DataFrame): DataFrame con ['x', 'y', 'end_x', 'end_y', 'is_progressive']
        ax: Matplotlib axis.
        pitch_obj: Objeto Pitch de mplsoccer.
        theme: 'dark' o 'light'.
    """
    if passes_df.empty:
        return
        
    t = THEMES[theme]
    
    x = passes_df.get('x', pd.Series(dtype=float))
    y = passes_df.get('y', pd.Series(dtype=float))
    end_x = passes_df.get('end_x', pd.Series(dtype=float))
    end_y = passes_df.get('end_y', pd.Series(dtype=float))
    
    # Supongamos que tenemos una bandera para pases progresivos o clave
    prog = passes_df.get('is_progressive', pd.Series([False]*len(passes_df)))
    
    # 1. Dibujar Pases Normales (Alta transparencia y más delgados para no hacer ruido)
    pitch_obj.arrows(
        x[~prog], y[~prog], end_x[~prog], end_y[~prog],
        width=1.5,
        headwidth=3, headlength=4, headaxislength=3,
        color=t['line_color'], 
        alpha=0.2, # <-- La clave para evitar el spaghetti plot
        ax=ax,
        label='Pase Regular'
    )
    
    # 2. Dibujar Pases Progresivos (Destacados y con mayor opacidad)
    pitch_obj.arrows(
        x[prog], y[prog], end_x[prog], end_y[prog],
        width=2.5,
        headwidth=4, headlength=5, headaxislength=4,
        color=t['home_color'],
        alpha=0.8,
        ax=ax,
        label='Pase Progresivo'
    )
    
def plot_heatmap(events_df, ax, pitch_obj, theme='dark'):
    """
    Dibuja un mapa de calor táctico (Kernel Density Estimate).
    
    Args:
        events_df (pd.DataFrame): DataFrame con ['x', 'y'].
        ax: Matplotlib axis.
        pitch_obj: Objeto Pitch de mplsoccer.
        theme: 'dark' o 'light'.
    """
    if events_df.empty:
        return
        
    x = events_df.get('x', pd.Series(dtype=float))
    y = events_df.get('y', pd.Series(dtype=float))
    
    # Seleccionamos colormap (magma es excelente para fondos oscuros)
    cmap = 'magma' if theme == 'dark' else 'Blues'
    
    # mplsoccer provee una función kdeplot que agrupa eventos en un mapa de calor
    pitch_obj.kdeplot(
        x, y,
        ax=ax,
        cmap=cmap,
        fill=True,
        levels=100,
        alpha=0.6,
        zorder=1 # Zorder bajo para que quede por debajo de otros puntos/tiros
    )
