import gvar as gv
import lsqfit
import numpy as np

#Superallowed beta decays
K=gv.gvar(8120.27648E-10, 2.6E-14) 
GF=gv.gvar(1.1663787E-5, 6.6E-12)
DELTA_R_superalowed=gv.gvar(0.02467, 0.00027)
A_superallowed = np.sqrt(K/(2*GF**2 * (1+DELTA_R_superalowed)))
Ft=gv.gvar(3072.24, 1.85)  #Ft value
B_superallowed = np.sqrt(Ft)

#Neutron decay
Delta_R_neutron=gv.gvar(0.03983, 0.00027)
#tau_n=gv.gvar(874.4, 0.6)
#gA=gv.gvar(1.2756, 0.0013)
tau_n=gv.gvar(877.83, 0.3)
gA=gv.gvar(1.27641, 0.00056)
factor_neutron=5099.33/(tau_n*(1+3*gA**2))
A_neutron= np.sqrt(factor_neutron)
B_neutron=np.sqrt(1+Delta_R_neutron)
#Factor_neutron=(1-3*gA.mean**2)/(1+3*gA.mean**2)

#Semileptonic kaon decays
A_semileptonic=gv.gvar(0.21656, 0.00035) #f_+(0)*|Vus|
B_semileptonic=gv.gvar(0.9698, 0.0017) #f+(0)

# Semileptonic kaon and pion decays (2)
A_semi_2=gv.gvar(0.22216, 0.00075505) #Vus*fK/Vud*fpi
#B_semileptonic=gv.gvar(0.9698, 0.0017) #fK(0)

# Leptonic Decays
A_Leptonic=gv.gvar(0.27679, 0.000344093) #fK/fpi*|Vus/Vud|
fk_fpi=gv.gvar(1.1978, 0.0022) #fK/fpi
B_Leptonic=fk_fpi
# Tau inclusive lattice
Rus= gv.gvar(0.1632, 0.0027)
A_tinclusive=np.sqrt(Rus)
ratio_tinclusive= gv.gvar(3.407,0.022) #Rus/(vud**2)
B_tinclusive=np.sqrt(ratio_tinclusive)


# Tau exclusive lattice
B=gv.gvar(0.0644,0.0009) #B10/9
A_texclusive= np.sqrt(B) 
delta_exclusive=gv.gvar(0.10,0.8)/100
m_tau=gv.gvar(1776.93, 0.09) 
m_kaon=gv.gvar(493.677, 0.015)
m_pion=gv.gvar(139.57039, 0.00018)
B_texclusive= fk_fpi*((m_tau**2 - m_kaon**2)/(m_tau**2 - m_pion**2))*np.sqrt(1+delta_exclusive)

# Tau OPE
Rud= gv.gvar(3.47000, 0.008)
Delta= gv.gvar(0.239, 0.032)

y_data = {
    'superallowed': A_superallowed/B_superallowed,
    'neutron':      A_neutron/B_neutron,
    'semileptonic': A_semileptonic/B_semileptonic,
    #'T-inclusive': A_tinclusive/B_tinclusive,
    'Leptonic': A_Leptonic/B_Leptonic,
    'semi_2': A_semi_2/B_semileptonic,
    'T-exclusive': A_texclusive/B_texclusive,
    #'Rus': Rus
}

# PRIORS 
priors = {
    # Incógnitas 
    'vud': gv.gvar(0.9, 1.0),
    'vus': gv.gvar(0.2, 1.0),
    'epsilon_BMS': gv.gvar(0.0, 0.01) 
}  


def modelo(p):
    return { 
        'superallowed': p['vud'], 
        'neutron':      p['vud'], 
        'semileptonic': p['vus']*(1+p['epsilon_BMS']), 
        #'T-inclusive':  p['vus'], 
        'Leptonic':     (p['vus']/p['vud'])*(1-p['epsilon_BMS']), 
        'semi_2': (p['vus']/p['vud'])*(1+p['epsilon_BMS']),
        'T-exclusive':   (p['vus']/p['vud'])*(1-p['epsilon_BMS']),
        #'Rus': (p['vus'])**2 * (Rud.mean / (p['vud']*(1-p['epsilon_BMS']))**2 - Delta.mean)
    }



# Ajuste
fit = lsqfit.nonlinear_fit(data=y_data, prior=priors, fcn=modelo)



print("RESULTADOS DEL AJUSTE GLOBAL CON LSQFIT\n")
print("-" * 45)

vud_fit = fit.p['vud']
vus_fit = fit.p['vus']


print(f"Vud = {vud_fit.mean:.5f} +/- {vud_fit.sdev:.5f}")
print(f"Vus = {vus_fit.mean:.5f} +/- {vus_fit.sdev:.5f}\n")
print(f"Epsilon_BMS = {fit.p['epsilon_BMS'].mean:.5f} +/- {fit.p['epsilon_BMS'].sdev:.5f}\n")


# Correlación y métricas
corr_matrix = gv.evalcorr([vud_fit, vus_fit])
print("MÉTRICAS DEL AJUSTE")
print("-" * 45)
print(f"Coeficiente de correlación (rho): {corr_matrix[0, 1]:.5f}")
print(f"Chi2/dof = {fit.chi2:.5f} / {fit.dof}\n")

print("DETALLES COMPLETOS DEL AJUSTE")
print("-" * 45)
print(fit)