import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 26,   # Nombre del eje (|Vud|) más grande
    'xtick.labelsize': 20,  # ¡NÚMEROS DEL EJE MÁS GRANDES!
})

colores_vud = [
    '#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', 
    '#9467BD', '#8C564B', '#008080', '#7F7F7F'
]

valores_Vud = []
errores_Vud = []
etiquetas_y = []
colores_id = []  # Cuarta columna para el identificador de color

# Lectura del fichero de datos
with open("datos_Vud.txt", "r", encoding="utf-8") as archivo:
    for linea in archivo:
        linea = linea.strip()
        if not linea or linea.startswith("#"): 
            continue
        partes = linea.split(",")
        # Comprobamos que tenga las 4 columnas requeridas
        if len(partes) >= 4:
            valores_Vud.append(float(partes[0].strip()))
            errores_Vud.append(float(partes[1].strip()))
            etiquetas_y.append(partes[2].strip())
            colores_id.append(partes[3].strip())

posiciones_y = list(range(len(etiquetas_y)))
posiciones_y.reverse() 

plt.figure(figsize=(10, 8)) 

# Buscar el dato superallowed para pintar la banda de fondo
vud_super = None
err_super = None
for v, e, lab in zip(valores_Vud, errores_Vud, etiquetas_y):
    if "superallowed" in lab.lower():
        vud_super = v
        err_super = e
        break

if vud_super is not None:
    plt.axvspan(vud_super - err_super, vud_super + err_super, 
                color='#673793', alpha=0.15, zorder=1, label='Banda Superallowed')

# Diccionario dinámico para asociar cada identificador a un color único
mapa_colores = {}

print("======= ASIGNACIÓN DE COLORES POR DATO (Vud) =======")

# Pintar los puntos experimentales y sus textos
for i in range(len(valores_Vud)):
    id_color = colores_id[i]
    
    # Si el texto de la columna es un código hexadecimal directo
    if id_color.startswith('#'):
        color_v = id_color
    else:
        # Si es una etiqueta de grupo/categoría
        if id_color not in mapa_colores:
            mapa_colores[id_color] = colores_vud[len(mapa_colores) % len(colores_vud)]
        color_v = mapa_colores[id_color]
    
    # IMPRESIÓN EN CONSOLA: Muestra qué color se le asigna a cada dato
    print(f"Dato: {etiquetas_y[i]:<20} | Grupo/ID del .txt: {id_color:<12} -> Color asignado: {color_v}")
    
    plt.errorbar(valores_Vud[i], posiciones_y[i], 
                 xerr=errores_Vud[i], 
                 fmt='o', color=color_v,
                 capsize=7, capthick=2, elinewidth=2, markersize=10, zorder=3)
    
    plt.text(valores_Vud[i], posiciones_y[i] - 0.35, 
             etiquetas_y[i], 
             color=color_v, ha='center', va='top', fontsize=23)

print("====================================================")

# Estética general
plt.yticks([]) 
plt.xlabel('$|V_{ud}|$', labelpad=15)
plt.grid(axis='x', linestyle=':', alpha=0.5, zorder=0)

plt.ylim(-1, len(etiquetas_y))
plt.tight_layout()

# Guardar y mostrar
plt.savefig("grafico_Vud.pdf", bbox_inches='tight')
plt.show()