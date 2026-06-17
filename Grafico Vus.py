import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. AJUSTE DE TAMAÑO: Formato idéntico al programa de Vud
mpl.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 26,   # Nombre del eje (|Vus|) más grande
    'xtick.labelsize': 17,  # ¡NÚMEROS DEL EJE MÁS GRANDES!
})

colores_vus = [
    '#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', 
    '#9467BD', '#8C564B', '#008080', '#7F7F7F'
]

valores_Vus = []
errores_Vus = []
etiquetas_y = []
colores_id = []  # Cuarta columna del archivo

# Lectura del fichero de datos
with open("datos_Vus.txt", "r", encoding="utf-8") as archivo:
    for linea in archivo:
        linea = linea.strip()
        if not linea or linea.startswith("#"): 
            continue
        partes = linea.split(",")
        if len(partes) >= 4:
            valores_Vus.append(float(partes[0].strip()))
            errores_Vus.append(float(partes[1].strip()))
            etiquetas_y.append(partes[2].strip())
            colores_id.append(partes[3].strip())

posiciones_y = list(range(len(etiquetas_y)))
posiciones_y.reverse() 

plt.figure(figsize=(10, 8)) 

# Buscar el dato semileptónico para pintar la banda de fondo
val_semi = None
err_semi = None
for v, e, lab in zip(valores_Vus, errores_Vus, etiquetas_y):
    if "l3" in lab or "kl3" in lab or "semi" in lab.lower() or "ell" in lab:
        val_semi = v
        err_semi = e
        break

if val_semi is None:
    val_semi, err_semi = 0.22330, 0.00053

if val_semi is not None:
    plt.axvspan(val_semi - err_semi, val_semi + err_semi, 
                color="#2C68A0", alpha=0.15, zorder=1, label='Banda Semileptónica')

# Diccionario dinámico para asociar cada identificador a un color único
mapa_colores = {}

print("======= ASIGNACIÓN DE COLORES POR DATO =======")

# Pintar los puntos experimentales y sus textos
for i in range(len(valores_Vus)):
    id_color = colores_id[i]
    
    # Si es código hexadecimal directo
    if id_color.startswith('#'):
        color_v = id_color
    else:
        # Si es un nombre de grupo/categoría
        if id_color not in mapa_colores:
            mapa_colores[id_color] = colores_vus[len(mapa_colores) % len(colores_vus)]
        color_v = mapa_colores[id_color]
    
    # IMPRESIÓN EN CONSOLA
    print(f"Dato: {etiquetas_y[i]:<20} | Grupo/ID del .txt: {id_color:<12} -> Color asignado: {color_v}")
    
    plt.errorbar(valores_Vus[i], posiciones_y[i], 
                 xerr=errores_Vus[i], 
                 fmt='o', color=color_v,
                 capsize=7, capthick=2, elinewidth=2, markersize=10, zorder=3)
    
    # =========================================================================
    # NUEVO CONTROL DE POSICIÓN DE TEXTO (Aislamos solo el tau average)
    # =========================================================================
    if "Average" in etiquetas_y[i] and "tau" in etiquetas_y[i].lower():
        # SOLO ESTE DATO: Se pone a la izquierda, restando el error y un margen
        x_texto = valores_Vus[i] - errores_Vus[i] - 0.0002
        y_texto = posiciones_y[i]
        alineacion_h = 'right'
        alineacion_v = 'center'
    else:
        # EL RESTO DEL GRÁFICO: Se mantiene igual que lo tenías
        x_texto = valores_Vus[i]
        alineacion_h = 'center'
        if "$\tau$" in etiquetas_y[i]:
            offset = 0.3
            alineacion_v = 'bottom'  # Corregido (antes tenías 'right')
        else:
            offset = -0.4
            alineacion_v = 'top'
        y_texto = posiciones_y[i] + offset
    
    # Pintamos el texto con las coordenadas calculadas
    plt.text(x_texto, y_texto, 
             etiquetas_y[i], 
             color=color_v, ha=alineacion_h, va=alineacion_v, fontsize=14.5)

print("==============================================")

# Estética general
plt.yticks([]) 
plt.xlabel('$|V_{us}|$', labelpad=15)
plt.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)

# GESTIÓN DEL ENCUADRE SIMÉTRICO
distancia_maxima = max(abs(v - val_semi) for v in valores_Vus)
semi_ancho_eje = distancia_maxima + 0.004  
plt.xlim(val_semi - semi_ancho_eje, val_semi + semi_ancho_eje)

# LÍNEA HORIZONTAL DIVISIONAL
if len(posiciones_y) >= 9:
    if len(posiciones_y) > 9:
        y_sep = (posiciones_y[8] + posiciones_y[9]) / 2
    else:
        y_sep = posiciones_y[8] - 0.5
        
    plt.axhline(y=y_sep, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    
    x_texto = (val_semi - semi_ancho_eje) + (semi_ancho_eje * 0.05)
    plt.text(x_texto, y_sep - 0.50, 'Asumiendo unitariedad', 
             color='black', ha='left', va='bottom', fontsize=14.5, zorder=5)

plt.ylim(-1, len(etiquetas_y))
plt.tight_layout()

# Guardar y mostrar
plt.savefig("grafico_Vus.pdf", bbox_inches='tight')
plt.show()