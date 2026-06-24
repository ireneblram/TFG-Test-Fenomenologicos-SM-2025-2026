import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker


mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 30,   # Nombre de los ejes (|Vud| y |Vus|)
    'xtick.labelsize': 18  # Números de los ejes grandes
})

colores_lista = [
    '#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', 
    '#9467BD', '#8C564B', '#008080', '#7F7F7F'
]


mapa_colores = {} 


def leer_datos(nombre_fichero):
    val, err, etiq, col_id = [], [], [], []
    try:
        with open(nombre_fichero, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.strip()
                if not linea or linea.startswith("#"): 
                    continue
                partes = linea.split(",")
                if len(partes) >= 4:
                    val.append(float(partes[0].strip()))
                    err.append(float(partes[1].strip()))
                    etiq.append(partes[2].strip())
                    col_id.append(partes[3].strip())
    except FileNotFoundError:
        print(f"⚠️ Aviso: No se encontró el archivo '{nombre_fichero}'.")
    return val, err, etiq, col_id


valores_Vud, errores_Vud, etiquetas_Vud, id_Vud = leer_datos("datos_Vud.txt")
valores_Vus, errores_Vus, etiquetas_Vus, id_Vus = leer_datos("datos_Vus.txt")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

def dibujar_panel(ax, valores, errores, etiquetas, colores_id, nombre_eje, palabra_banda, color_banda):
    if not valores:
        return
        
    posiciones_y = list(range(len(etiquetas)))
    posiciones_y.reverse()

    # Buscar el dato de referencia para pintar la banda de fondo
    v_ref, e_ref = None, None
    for v, e, lab in zip(valores, errores, etiquetas):
        if palabra_banda.lower() in lab.lower():
            v_ref = v
            e_ref = e
            break

    if v_ref is not None:
        ax.axvspan(v_ref - e_ref, v_ref + e_ref, 
                   color=color_banda, alpha=0.15, zorder=1, label=f'Banda {palabra_banda}')

    print(f"--- Asignación de colores para {nombre_eje} ---")
    
    # Pintar los puntos y sus textos
    for i in range(len(valores)):
        id_color = colores_id[i]
        
        if id_color.startswith('#'):
            color_v = id_color
        else:
            if id_color not in mapa_colores:
                mapa_colores[id_color] = colores_lista[len(mapa_colores) % len(colores_lista)]
            color_v = mapa_colores[id_color]
            
        print(f"Dato: {etiquetas[i]:<20} | ID: {id_color:<10} -> Color: {color_v}")

        # Dibujar punto con barras de error
        ax.errorbar(valores[i], posiciones_y[i], 
                    xerr=errores[i], fmt='o', color=color_v,
                    capsize=7, capthick=2, elinewidth=2, markersize=10, zorder=3)
        
        # Escribir la etiqueta justo debajo del punto
        ax.text(valores[i], posiciones_y[i] - 0.35, 
                etiquetas[i], 
                color=color_v, ha='center', va='top', fontsize=17)


    ax.set_yticks([]) 
    ax.set_xlabel(nombre_eje, labelpad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)
    ax.set_ylim(-1, len(etiquetas))



# Dibujamos Vud en el panel izquierdo (ax1) buscando la palabra 'superallowed'
dibujar_panel(ax1, valores_Vud, errores_Vud, etiquetas_Vud, id_Vud, 
              nombre_eje='$|V_{ud}|$', palabra_banda='superallowed', color_banda='#673793')

# Dibujamos Vus en el panel derecho (ax2) buscando la palabra 'semileptonic' (cámbiala por la que necesites)
dibujar_panel(ax2, valores_Vus, errores_Vus, etiquetas_Vus, id_Vus, 
              nombre_eje='$|V_{us}|$', palabra_banda='semileptonic', color_banda='#1F77B4')


plt.tight_layout()
plt.savefig("grafico_Vud_Vus_Paralelo.pdf", bbox_inches='tight')
plt.show()
