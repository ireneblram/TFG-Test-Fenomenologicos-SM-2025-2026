import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import AutoMinorLocator


mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 22,   # Etiquetas de los ejes grandes
    'xtick.labelsize': 16,  # Números del eje X grandes
    'ytick.labelsize': 16,  # Números del eje Y grandes
})

# Fichero de entrada
fichero_datos = 'datos_chi2_rho.txt'


color_sin = '#4361ee'  
color_con = '#f76425'  
color_top = '#80ce68'  #


try:
    # Leemos las 3 columnas (Chi2/dof, rho, con_leptonica)
    datos = np.loadtxt(fichero_datos)
    chi2_dof = datos[:, 0]
    rho = datos[:, 1]
    con_leptonica = datos[:, 2]  # 0 = sin, 1 = con (parcial), 2 = completo
    
except Exception as e:
    print(f"Error al leer el fichero {fichero_datos}: {e}")
    print("Cargando valores de ejemplo por seguridad con tu convención (0, 1, 2)...")
    chi2_dof = np.array([1.2, 1.8, 2.5, 3.2, 3.8])
    rho = np.array([0.01, 0.03, 0.11, 0.13, 0.13])
    con_leptonica = np.array([0, 0, 1, 1, 2])


fig, ax = plt.subplots(figsize=(10, 7))


ax.axvspan(0.7, 1.3, color="#2e4ac4", alpha=0.1, label=r'Región ideal ($\chi^2/\mathrm{dof} \sim 1$)', zorder=1)
ax.axvline(1, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=2)


ax.axhspan(-0.02, 0.02, color="#b74a4a", alpha=0.12, label=r'Independencia estadística ($\rho \sim 0$)', zorder=1)
ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=2)


mask_sin = (con_leptonica == 0)
mask_con = (con_leptonica == 1)
mask_top = (con_leptonica == 2)


if np.any(mask_sin):
    ax.scatter(chi2_dof[mask_sin], rho[mask_sin], color=color_sin, s=120, 
               edgecolor='black', linewidth=0.8, marker='o', 
               alpha=0.9, label=r'Procesos nucleares, semileptónicos y física del $\tau$', zorder=4)


if np.any(mask_con):
    ax.scatter(chi2_dof[mask_con], rho[mask_con], color=color_con, s=120, 
               edgecolor='black', linewidth=0.8, marker='s', 
               alpha=0.9, label='Inclusión adicional del canal leptónico', zorder=4)


if np.any(mask_top):
    x_top = chi2_dof[mask_top][0]
    y_top = rho[mask_top][0]
    

    ax.scatter(x_top, y_top, color=color_top, s=320, 
               edgecolor='black', linewidth=1.0, marker='*', 
               label='Ajuste Global Definitivo (Dataset completo)', zorder=5)



ax.xaxis.set_minor_locator(AutoMinorLocator(2))  
ax.yaxis.set_minor_locator(AutoMinorLocator(2))  
ax.tick_params(which='minor', length=4, color='lightgray')  
ax.tick_params(which='major', length=7)


ax.set_xlabel(r'$\chi^2 / \mathrm{dof}$', fontsize=20, labelpad=15)
ax.set_ylabel(r'$\rho(V_{ud}, V_{us})$', fontsize=20, labelpad=15)



ax.grid(True, linestyle=':', alpha=0.4, zorder=0)


ax.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12, loc='upper left')

ax.set_xlim(0, np.max(chi2_dof) * 1.15)
ax.set_ylim(np.min(rho) - 0.04, np.max(rho) * 1.25)

# Guardar en PDF de alta calidad
plt.tight_layout()
plt.savefig('rho_vs_chi2dof_combinado.pdf', format='pdf', bbox_inches='tight')

print("\n¡Gráfico definitivo generado con éxito sin líneas sobrantes!: 'rho_vs_chi2dof_combinado.pdf'")
plt.show()
