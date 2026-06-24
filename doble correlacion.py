import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib as mpl


# DATOS INDEPENDIENTES 
# Superallowed beta decays
K = gv.gvar(8120.27648E-10, 2.6E-14) 
GF = gv.gvar(1.1663787E-5, 6.6E-12)
DELTA_R_superalowed = gv.gvar(0.02467, 0.00027)
A_superallowed = np.sqrt(K / (2 * GF**2 * (1 + DELTA_R_superalowed)))
Ft = gv.gvar(3072.24, 1.85)  
B_superallowed = np.sqrt(Ft)

# Neutron decay
Delta_R_neutron = gv.gvar(0.03983, 0.00027)
tau_n = gv.gvar(874.4, 0.6)
gA = gv.gvar(1.2756, 0.0013)
factor_neutron = 5099.33 / (tau_n * (1 + 3 * gA**2))
A_neutron = np.sqrt(factor_neutron)
B_neutron = np.sqrt(1 + Delta_R_neutron)

# Tau inclusive lattice
Rus = gv.gvar(0.1632, 0.0027)
A_tinclusive = np.sqrt(Rus)
ratio_tinclusive = gv.gvar(3.407, 0.022) 
B_tinclusive = np.sqrt(ratio_tinclusive)

# Tau exclusive lattice
A_texclusive = np.sqrt(gv.gvar(0.0644, 0.0009)) 
delta_exclusive = gv.gvar(0.10, 0.8) / 100
m_tau = gv.gvar(1776.93, 0.09) 
m_kaon = gv.gvar(493.677, 0.015)
m_pion = gv.gvar(139.57039, 0.00018)

# Tau OPE
Rud = gv.gvar(3.47000, 0.008)
Delta = gv.gvar(0.239, 0.032)

# Parámetros base para las matrices
mean_A_semi, sigma_A_semi   = 0.21656, 0.00035
mean_A_lep,  sigma_A_lep    = 0.27679, 0.000344093
mean_A_semi2, sigma_A_semi2 = 0.22216, 0.00075505

mean_B_semi, sigma_B_semi   = 0.9698, 0.0017
mean_B_lep,  sigma_B_lep    = 1.1978, 0.0022 

# Priors y modelo de lsqfit
priors = {'vud': gv.gvar(0.9, 1.0), 'vus': gv.gvar(0.2, 1.0)}  
def modelo(p):
    return { 
        'superallowed': p['vud'], 
        'neutron':      p['vud'], 
        'semileptonic': p['vus'], 
        'T-inclusive':  p['vus'], 
        'Leptonic':     (p['vus'] / p['vud']), 
        'semi_2':       (p['vus'] / p['vud']),
        'T-exclusive':  (p['vus'] / p['vud']),
        'Rus':          p['vus']**2 * (Rud.mean / p['vud']**2 - Delta.mean)
    }


# CORRELACIÓN TEÓRICA vs EXPERIMENTAL

N = 40 
rho_A_vals = np.linspace(-1, 1, N) # Eje X (Experimental)
rho_B_vals = np.linspace(-1, 1, N) # Eje Y (Teórica)

# Matrices para almacenar los resultados en formato de cuadrícula (grid)
sigmas_matrix = np.zeros((N, N))

print(f"Ejecutando {N*N} ajustes para el mapa 2D...")

for i, r_A in enumerate(rho_A_vals):
    for j, r_B in enumerate(rho_B_vals):
        
        #  Sector A 
        cov_A_sl_lep = r_A * sigma_A_semi * sigma_A_lep
        cov_A_sl_s2  = 1.0 * sigma_A_semi * sigma_A_semi2
        cov_A_lep_s2 = r_A * sigma_A_lep * sigma_A_semi2

        cov_matrix_A = np.array([
            [sigma_A_semi**2, cov_A_sl_lep,     cov_A_sl_s2],
            [cov_A_sl_lep,    sigma_A_lep**2,   cov_A_lep_s2],
            [cov_A_sl_s2,     cov_A_lep_s2,     sigma_A_semi2**2]
        ])
        A_semileptonic, A_Leptonic, A_semi_2 = gv.gvar([mean_A_semi, mean_A_lep, mean_A_semi2], cov_matrix_A)

        # Sector B 
        cov_B_sl_lep = r_B * sigma_B_semi * sigma_B_lep
        cov_matrix_B = np.array([
            [sigma_B_semi**2, cov_B_sl_lep],
            [cov_B_sl_lep,    sigma_B_lep**2]
        ])
        B_semileptonic, B_Leptonic = gv.gvar([mean_B_semi, mean_B_lep], cov_matrix_B)
        
        # Actualizamos variables dependientes
        B_texclusive = B_Leptonic * ((m_tau**2 - m_kaon**2) / (m_tau**2 - m_pion**2)) * np.sqrt(1 + delta_exclusive)

        y_data = {
            'superallowed': A_superallowed / B_superallowed, 'neutron': A_neutron / B_neutron,
            'semileptonic': A_semileptonic / B_semileptonic, 'T-inclusive': A_tinclusive / B_tinclusive,
            'Leptonic': A_Leptonic / B_Leptonic, 'semi_2': A_semi_2 / B_semileptonic,
            'T-exclusive': A_texclusive / B_texclusive, 'Rus': Rus
        }

       
        try:
            fit = lsqfit.nonlinear_fit(data=y_data, prior=priors, fcn=modelo, svdcut=1e-3)
            vud_res, vus_res = fit.p['vud'], fit.p['vus']
            
            # Reconstruimos la matriz para propagar Delta
            corr_ajuste = gv.evalcorr([vud_res, vus_res])[0, 1]
            cov_cruzada = corr_ajuste * vud_res.sdev * vus_res.sdev
            matriz_vud_vus = np.array([[vud_res.sdev**2, cov_cruzada], [cov_cruzada, vus_res.sdev**2]])
            
            vud_gv, vus_gv = gv.gvar([vud_res.mean, vus_res.mean], matriz_vud_vus)
            delta_gv = 1 - vud_gv**2 - vus_gv**2
            
            # Guardamos la sigma. OJO: j es fila (Y), i es columna (X)
            sigmas_matrix[j, i] = delta_gv.mean / delta_gv.sdev
        except:
            # Si el ajuste falla por matrices matemáticamente imposibles, asignamos NaN
            sigmas_matrix[j, i] = np.nan

print("Ajustes completados. Generando gráfico...")


mpl.rcParams.update({'font.size': 11, 'axes.labelsize': 13})
fig, ax = plt.subplots(figsize=(8, 6.5))

# Creamos la malla para los ejes
X, Y = np.meshgrid(rho_A_vals, rho_B_vals)


niveles_color = np.linspace(1.0, 5.0, 50)

 
# El parámetro extend='max' le dice que todo lo que supere el 5.0 lo pinte del rojo máximo y lo agrupe ahí.
mapa = ax.contourf(X, Y, sigmas_matrix, 
                   levels=niveles_color, 
                   cmap='Spectral_r', 
                   extend='max')

# Añadimos líneas negras sólidas indicando las "fronteras" de 1, 2 y 3 sigmas
lineas = ax.contour(X, Y, sigmas_matrix, levels=[1.5, 2, 3], colors='black', linestyles='dashed', linewidths=1.2)
ax.clabel(lineas, inline=True, fontsize=10, fmt='%1.0f $\sigma$')

# Línea diagonal rho_A = rho_B 
ax.plot([-1, 1], [-1, 1], color="#649dc6", linestyle=':', lw=2, label=r'$\rho_{th} = \rho_{exp}$')

# Barra de color a la derecha
cbar = fig.colorbar(mapa, ax=ax, pad=0.03)
cbar.set_label(r'Tensión de Unitariedad ($\sigma_{\mathrm{CKM}}$)', rotation=270, labelpad=20, fontsize=16)

#ax.set_title('Sensibilidad Bidimensional a las Correlaciones $\\rho_{exp}$ y $\\rho_{th}$', pad=15, fontweight='bold')
ax.set_xlabel(r'Correlación Experimental  ($\rho_{exp}$)', fontsize=16)
ax.set_ylabel(r'Correlación Teórica  ($\rho_{th}$)', fontsize=16)
ax.legend(loc='lower left', framealpha=0.9)


ax.fill_between(x=[-1, 0], y1=[0, 0], y2=[1, 1], color="#C5BFBF5C", alpha=0.35, zorder=1)


ax.fill_between(x=[0, 1], y1=[-1, -1], y2=[0, 0], color="#C5BFBF5C", alpha=0.35, zorder=1)


ax.text(0.25, 0.9, 'Correlaciones\nPositivas', 
        color='black', alpha=0.9, ha='center', va='center', fontsize=13, zorder=2)   

ax.text(-0.75, -0.1, 'Correlaciones\n Negativas', 
        color='black', alpha=0.9, ha='center', va='center', fontsize=13, zorder=2)

# Opcional: Ejes centrales para delimitar bien la cruz del cero
ax.axhline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.5)
ax.axvline(0, color='black', linestyle='-', linewidth=0.6, alpha=0.5)

plt.tight_layout()
plt.savefig('mapa_correlaciones_2D.pdf', format='pdf')
plt.show()
