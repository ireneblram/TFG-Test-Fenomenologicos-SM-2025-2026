import gvar as gv
import lsqfit
import numpy as np

# ==============================================================================
# 1. DATOS INDEPENDIENTES (Constantes fijas fuera del bucle para optimizar)
# ==============================================================================
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

# Parámetros experimentales y teóricos base para las matrices de covarianza
mean_A_semi, sigma_A_semi   = 0.21656, 0.00035
mean_A_lep,  sigma_A_lep    = 0.27679, 0.000344093
mean_A_semi2, sigma_A_semi2 = 0.22216, 0.00075505

mean_B_semi, sigma_B_semi   = 0.9698, 0.0017
mean_B_lep,  sigma_B_lep    = 1.1978, 0.0022 

# ==============================================================================
# 2. CONFIGURACIÓN DEL ANÁLISIS DE SENSIBILIDAD (Bucle rho)
# ==============================================================================
N = 100
valores_rho = np.linspace(-1, 1, N)

# Listas vacías para almacenar la evolución de los resultados
vud_medias, vud_errores = [], []
vus_medias, vus_errores = [], []
chi2_valores, rho_ajuste = [], []

# Definición de priors y modelo (Fijos)
priors = {
    'vud': gv.gvar(0.9, 1.0),
    'vus': gv.gvar(0.2, 1.0)
}  

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

print(f"Ejecutando {N} ajustes correlacionados secuenciales...")

# EL BUCLE: Ahora todo lo que depende de rho_valores se calcula en cada iteración
for i in range(N):
    rho_actual = valores_rho[i]
    
    # --- Sector A (Correlaciones Teóricas/Experimentales cruzadas) ---
    rho_A_semi_lep = rho_actual   
    rho_A_semi_semi2 = 1.0        
    rho_A_lep_semi2 = rho_actual        

    cov_A_sl_lep = rho_A_semi_lep * sigma_A_semi * sigma_A_lep
    cov_A_sl_s2  = rho_A_semi_semi2 * sigma_A_semi * sigma_A_semi2
    cov_A_lep_s2 = rho_A_lep_semi2 * sigma_A_lep * sigma_A_semi2

    cov_matrix_A = np.array([
        [sigma_A_semi**2, cov_A_sl_lep,     cov_A_sl_s2],
        [cov_A_sl_lep,    sigma_A_lep**2,   cov_A_lep_s2],
        [cov_A_sl_s2,     cov_A_lep_s2,     sigma_A_semi2**2]
    ])

    A_semileptonic, A_Leptonic, A_semi_2 = gv.gvar(
        [mean_A_semi, mean_A_lep, mean_A_semi2], 
        cov_matrix_A
    )

    # --- Sector B (Correlaciones de Lattice QCD o Factores de forma) ---
    rho_B_semi_lep = rho_actual  
    cov_B_sl_lep = rho_B_semi_lep * sigma_B_semi * sigma_B_lep
    
    cov_matrix_B = np.array([
        [sigma_B_semi**2, cov_B_sl_lep],
        [cov_B_sl_lep,    sigma_B_lep**2]
    ])

    B_semileptonic, B_Leptonic = gv.gvar(
        [mean_B_semi, mean_B_lep], 
        cov_matrix_B
    )
    
    # Reevaluamos B_texclusive dinámicamente con el B_Leptonic (fk_fpi) correlacionado
    fk_fpi = B_Leptonic 
    B_texclusive = fk_fpi * ((m_tau**2 - m_kaon**2) / (m_tau**2 - m_pion**2)) * np.sqrt(1 + delta_exclusive)

    # --- Construcción del Dataset Correlacionado para esta iteración ---
    y_data = {
        'superallowed': A_superallowed / B_superallowed,
        'neutron':      A_neutron / B_neutron,
        'semileptonic': A_semileptonic / B_semileptonic,
        'T-inclusive':  A_tinclusive / B_tinclusive,
        'Leptonic':     A_Leptonic / B_Leptonic,
        'semi_2':       A_semi_2 / B_semileptonic,
        'T-exclusive':  A_texclusive / B_texclusive,
        'Rus':          Rus
    }

    # --- Ejecución del ajuste no lineal bayesiano ---
    fit = lsqfit.nonlinear_fit(data=y_data, prior=priors, fcn=modelo)

    # --- Extracción y guardado de resultados ---
    vud_res = fit.p['vud']
    vus_res = fit.p['vus']
    
    vud_medias.append(vud_res.mean)
    vud_errores.append(vud_res.sdev)
    vus_medias.append(vus_res.mean)
    vus_errores.append(vus_res.sdev)
    
    chi2_valores.append(fit.chi2/fit.dof)
    # Correlación final calculada entre Vud y Vus por lsqfit
    corr_matrix = gv.evalcorr([vud_res, vus_res])
    rho_ajuste.append(corr_matrix[0, 1])

print("¡Proceso completado con éxito!\n")

# ==============================================================================
# 3. MUESTRA DE RESULTADOS SELECCIONADOS (Puntos críticos del escrutinio)
# ==============================================================================
print(f"{'rho_input':^10} | {'Vud central':^12} +/- {'Error Vud':<9} | {'Vus central':^12} +/- {'Error Vus':<9} | {'Chi2':^7}")
print("-" * 75)

# Te muestro una muestra representativa (cada 10 pasos) para no saturar la pantalla
for idx in range(0, N, 10):
    print(f"{valores_rho[idx]:10.2f} | {vud_medias[idx]:12.5f} +/- {vud_errores[idx]:<.5f} | {vus_medias[idx]:12.5f} +/- {vus_errores[idx]:<.5f} | {chi2_valores[idx]:7.2f}")

# Muestra el último punto de la lista (rho = 1.00)
print(f"{valores_rho[-1]:10.2f} | {vud_medias[-1]:12.5f} +/- {vud_errores[-1]:<.5f} | {vus_medias[-1]:12.5f} +/- {vus_errores[-1]:<.5f} | {chi2_valores[-1]:7.2f}")
# ==============================================================================
# 4. GUARDADO AUTOMÁTICO DE LOS RESULTADOS EN UN FICHERO .TXT
# ==============================================================================
nombre_fichero = "analisis__rho_chi2.txt"

with open(nombre_fichero, "w", encoding="utf-8") as f:
    # Encabezado formal con metadatos para tu TFG
    f.write("# =========================================================================\n")
    f.write("# RESULTADOS DEL ANÁLISIS DE SENSIBILIDAD - AJUSTE GLOBAL CKM (|Vud| y |Vus|)\n")
    f.write("# =========================================================================\n")
    f.write("# Columnas:\n")
    f.write("# 1. rho_input  : Coeficiente de correlación introducido en las matrices de covarianza.\n")
    f.write("# 2. Vud_mean   : Valor central (media) obtenido para |Vud|.\n")
    f.write("# 3. Vud_sdev   : Incertidumbre (desviación estándar) de |Vud|.\n")
    f.write("# 4. Vus_mean   : Valor central (media) obtenido para |Vus|.\n")
    f.write("# 5. Vus_sdev   : Incertidumbre (desviación estándar) de |Vus|.\n")
    f.write("# 6. rho_ajuste : Coeficiente de correlación final entre |Vud| y |Vus| (lsqfit).\n")
    f.write("# 7. chi2        : Valor de chi-cuadrado/ grados de libertad del ajuste.\n")
    f.write("# =========================================================================\n")
    
    # Cabecera de las columnas perfectamente alineadas
    f.write(f"{'rho_input':^12}\t{'Vud_mean':^12}\t{'Vud_sdev':^12}\t{'Vus_mean':^12}\t{'Vus_sdev':^12}\t{'rho_ajuste':^12}\t{'chi2':^12}\n")
    f.write("#" + "-"*95 + "\n")
    
    # Escritura indexada de los 100 puntos calculados con alta precisión
    for i in range(N):
        f.write(f"{valores_rho[i]:12.6f}\t"
                f"{vud_medias[i]:12.6f}\t"
                f"{vud_errores[i]:12.6f}\t"
                f"{vus_medias[i]:12.6f}\t"
                f"{vus_errores[i]:12.6f}\t"
                f"{rho_ajuste[i]:12.6f}\t"
                f"{chi2_valores[i]:12.6f}\n")

print(f"\n[INFO] Datos de sensibilidad exportados correctamente a: '{nombre_fichero}'")
