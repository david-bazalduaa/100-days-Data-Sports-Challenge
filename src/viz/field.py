"""
Day 13: NFL Visualization Library

Objective:
Create field plotting functions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Reutilizamos la misma paleta que en fútbol para mantener consistencia visual
THEMES = {
    'dark': {
        'fig_color': '#141414',
        'field_color': '#1e1e1e',
        'endzone_color': '#2a2a2a',
        'line_color': '#ffffff',
        'text_color': '#ffffff',
        'home_color': '#00d2ff',
        'away_color': '#ff0055',
    },
    'light': {
        'fig_color': '#f3f4f6',
        'field_color': '#ffffff',
        'endzone_color': '#e2e8f0',
        'line_color': '#cbd5e1',
        'text_color': '#1e1e1e',
        'home_color': '#0369a1',
        'away_color': '#be123c',
    }
}

def draw_field(theme='dark'):
    """
    Dibuja un campo de la NFL matemáticamente exacto usando parches de Matplotlib.
    Dimensiones: 120 yardas de largo x 53.3 yardas de ancho.
    
    Args:
        theme (str): 'dark' o 'light'
        
    Returns:
        fig, ax: Objetos de Matplotlib.
    """
    if theme not in THEMES:
        theme = 'dark'
    t = THEMES[theme]
    
    # 1. Crear figura y ejes
    fig, ax = plt.subplots(figsize=(12, 5.33)) # Relación de aspecto natural del campo
    fig.patch.set_facecolor(t['fig_color'])
    ax.set_facecolor(t['field_color'])
    
    # Desactivamos los bordes y ejes predeterminados de Matplotlib
    ax.axis('off')
    
    # Límites exactos del campo
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    
    # 2. Dibujar las Endzones (con contraste de color)
    # Endzone Izquierda (0 a 10)
    left_ez = patches.Rectangle((0, 0), 10, 53.3, facecolor=t['endzone_color'], edgecolor=t['line_color'], zorder=1)
    ax.add_patch(left_ez)
    # Endzone Derecha (110 a 120)
    right_ez = patches.Rectangle((110, 0), 10, 53.3, facecolor=t['endzone_color'], edgecolor=t['line_color'], zorder=1)
    ax.add_patch(right_ez)
    
    # Líneas de contorno exterior
    border = patches.Rectangle((0, 0), 120, 53.3, fill=False, edgecolor=t['line_color'], linewidth=2, zorder=2)
    ax.add_patch(border)
    
    # 3. Dibujar las líneas de yardas y los números
    # Matemáticas: Los números en NFL se pintan a 12 yardas de la banda (sideline)
    y_bottom_numbers = 12
    y_top_numbers = 53.3 - 12
    
    # Hash marks de la NFL están a 70 pies 9 pulgadas (~23.58 yardas) de cada banda
    y_bottom_hash = 23.58
    y_top_hash = 53.3 - 23.58
    
    # Iteramos cada 5 yardas entre la 10 y la 110 (zona de juego)
    for x in range(10, 111, 5):
        # Línea cruzada de lado a lado
        ax.plot([x, x], [0, 53.3], color=t['line_color'], alpha=0.3, linewidth=1, zorder=2)
        
        # Cada 10 yardas ponemos el número y un hash fuerte
        if x % 10 == 0 and 10 < x < 110:
            # Determinamos el número real de yarda para imprimir (va de 10 a 50 y baja a 10)
            if x <= 60:
                yard_num = x - 10
            else:
                yard_num = 110 - x
                
            # Pintar números inferiores
            ax.text(x, y_bottom_numbers, str(yard_num), color=t['line_color'], 
                    ha='center', va='center', fontsize=16, weight='bold', alpha=0.5)
            # Pintar números superiores (rotados 180 grados para la cámara de TV)
            ax.text(x, y_top_numbers, str(yard_num), color=t['line_color'], 
                    ha='center', va='center', fontsize=16, weight='bold', alpha=0.5, rotation=180)
            
        # Dibujar marcas de hash (cada 5 yardas)
        ax.plot([x - 0.5, x + 0.5], [y_bottom_hash, y_bottom_hash], color=t['line_color'], alpha=0.6, lw=1, zorder=2)
        ax.plot([x - 0.5, x + 0.5], [y_top_hash, y_top_hash], color=t['line_color'], alpha=0.6, lw=1, zorder=2)

    return fig, ax

def plot_play(play_df, ax, theme='dark'):
    """
    Dibuja trayectorias de jugadores o el balón para una jugada usando NFL tracking data.
    
    Args:
        play_df (pd.DataFrame): DataFrame con la trayectoria (mínimo: 'x', 'y', 'team', 'displayName')
        ax: Matplotlib axis retornado por draw_field.
        theme: 'dark' o 'light'
    """
    if play_df.empty:
        return
        
    t = THEMES.get(theme, THEMES['dark'])
    
    # Detectamos los equipos
    teams = play_df.get('team', pd.Series(['Unknown'])).unique()
    
    for team in teams:
        team_data = play_df[play_df['team'] == team]
        
        # Asignar color dinámico (excluyendo el balón)
        if team.lower() == 'football':
            color = '#8b4513' # Marrón
        else:
            color = t['home_color'] if team == teams[0] else t['away_color']
            
        # Agrupar por jugador para trazar su línea independiente
        if 'displayName' in team_data.columns:
            for player, player_data in team_data.groupby('displayName'):
                x = player_data['x'].values
                y = player_data['y'].values
                
                if team.lower() == 'football':
                    # Línea punteada para el balón
                    ax.plot(x, y, color=t['text_color'], linewidth=2.5, linestyle='--', zorder=4)
                    ax.scatter(x[-1], y[-1], color=color, s=60, zorder=5) # Posición final
                else:
                    # Línea semitransparente para la ruta del jugador
                    ax.plot(x, y, color=color, alpha=0.5, linewidth=1.5, zorder=3)
                    # Punto sólido para el jugador en su posición final
                    ax.scatter(x[-1], y[-1], color=color, edgecolors=t['text_color'], s=40, zorder=4)
