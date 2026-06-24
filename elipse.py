import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker


mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 24,   # Nombre de los ejes grande
    'xtick.labelsize': 16,  # Números del eje X grandes
    'ytick.labelsize': 16,  # Números del eje Y grandes   
})

# Función que calcula los parámetros de la elipse de confianza
def dibujar_elipse(ax, x_mean, y_mean, sig_x, sig_y, rho, sigma_level=1, **kwargs):
    cov = np.array([
        [sig_x**2, rho * sig_x * sig_y],
        [rho * sig_x * sig_y, sig_y**2]
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    if sigma_level == 1:
        escala = np.sqrt(2.30)  # 68.27% CL
    elif sigma_level == 2:
        escala = np.sqrt(6.18)  # 95.45% CL
    else:
        escala = 1.0
        
    width, height = 2 * escala * np.sqrt(eigenvalues)
    
    elipse = Ellipse(xy=(x_mean, y_mean), width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(elipse)
    return elipse


# DATOS
Vud_ajuste =  0.97381
Vud_error  =  0.00089
Vus_ajuste =  0.22411 
Vus_error  =  0.00041
correlacion_rho =  0.49527
chi2_dof = 0.563145
epsilon_s= -0.00372 
error_epsilon_s= 0.00088
epsilon_d=  0.00002
error_epsilon_d= 0.00088
#delta=-0.00370 
#error_delta= 0.00157

# Datos Experimentales (Bandas)
vud_exp = 0.97367
vud_err = 0.00032

vus_exp = 0.2233
vus_err = 0.00053

ratio_exp = 0.23108 
ratio_err = 0.00051 



# GRÁFICO 
fig, ax = plt.subplots(figsize=(8, 7))


factor_zoom_x = 2.2 * Vud_error
factor_zoom_y = 3.5 * Vus_error

lim_inf_x = Vud_ajuste - factor_zoom_x
lim_sup_x = Vud_ajuste + factor_zoom_x

lim_inf_y = Vus_ajuste - factor_zoom_y
lim_sup_y = Vus_ajuste + factor_zoom_y

ax.set_xlim(lim_inf_x, lim_sup_x)
ax.set_ylim(lim_inf_y, lim_sup_y)
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
vud_rango = np.linspace(lim_inf_x, lim_sup_x, 300)

# Dibujo de las bandas experimentales
ax.axvspan(vud_exp - vud_err, vud_exp + vud_err, 
           color='#ff7f0e', alpha=0.15, label=r'$\beta\to\beta$ $V_{ud}$', zorder=1)

ax.axhspan(vus_exp - vus_err, vus_exp + vus_err, 
           color='#2ca02c', alpha=0.15, label=r'$K\to \pi\ell\nu$ $V_{us}$', zorder=1)

vus_ratio_up = (ratio_exp + ratio_err) * vud_rango
vus_ratio_down = (ratio_exp - ratio_err) * vud_rango
ax.fill_between(vud_rango, vus_ratio_down, vus_ratio_up, 
                color='#9467bd', alpha=0.15, label=r'$K\to \ell\nu / \pi\to\ell\nu$ $V_{us}/V_{ud}$', zorder=1)

# Línea de unitaridad (negra continua)
vus_unitaridad = np.sqrt(1 - vud_rango**2)
ax.plot(vud_rango, vus_unitaridad, 'k-', linewidth=1.5, label=r'Unitariedad ($|V_{ud}|^2 + |V_{us}|^2 = 1$)', zorder=2)


dibujar_elipse(ax, Vud_ajuste, Vus_ajuste, Vud_error, Vus_error, correlacion_rho, 
               sigma_level=2, edgecolor="#436E53", facecolor='#436E53', alpha=0.10, 
               label=r'Ajuste Global $2\sigma$', zorder=3)

dibujar_elipse(ax, Vud_ajuste, Vus_ajuste, Vud_error, Vus_error, correlacion_rho, 
               sigma_level=1, edgecolor="#436E53", facecolor='#436E53', alpha=0.38, 
               label=r'Ajuste Global $1\sigma$', zorder=4)

ax.plot(Vud_ajuste, Vus_ajuste, 'ko', markersize=5, zorder=5)

# Nombres de los ejes
ax.set_xlabel('$V_{ud}$', labelpad=12)
ax.set_ylabel('$V_{us}$', labelpad=12)


# LEYENDA Y CUADRO 

texto_recuadro_datos = (
    f"Resultados del Ajuste Global:\n"
    f"$|V_{{ud}}| = {Vud_ajuste:.5f} \\pm {Vud_error:.5f}$\n"
    f"$|V_{{us}}| = {Vus_ajuste:.5f} \\pm {Vus_error:.5f}$\n"
    f"$\\rho = {correlacion_rho:.3f}$\n"
    f"$\\chi^2/\\text{{dof}} = {chi2_dof:.3f}$\n"
    f"$\\epsilon_d = {epsilon_d:.5f} \\pm {error_epsilon_d:.5f}$\n"
    f"$\\epsilon_s = {epsilon_s:.5f} \\pm {error_epsilon_s:.5f}$"
)

ax.text(
    -0.08, 1.45,                            
    texto_recuadro_datos,                  
    transform=ax.transAxes,               
    fontsize=15, 
    verticalalignment='top',              
    horizontalalignment='left', 
    linespacing=1.5,           
    bbox=dict(
        facecolor='white',                
        alpha=0.90, 
        edgecolor="#436E53",              
        boxstyle='round,pad=0.3',
        linewidth=1.0
    ),
    zorder=6
)


handles, labels = ax.get_legend_handles_labels()
leyenda = ax.legend(
    handles, labels, 
    loc='lower right',                    
    bbox_to_anchor=(1.1, 1.04),                                 
    fontsize=15, 
    framealpha=1.0, 
    edgecolor='#436E53',
    shadow=False
)
leyenda.set_zorder(6)

ax.grid(True, linestyle=':', alpha=0.5)
plt.subplots_adjust(top=0.85)

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color("#010101")
    spine.set_linewidth(1.0)

plt.savefig('elipse_2_parametro_DyS_editado.pdf', format='pdf', bbox_inches='tight')
print("\nGráfico optimizado con éxito. Archivo guardado: 'elipse_2_parametro_DyS_editado.pdf'")
plt.show()
