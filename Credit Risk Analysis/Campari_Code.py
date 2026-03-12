import pandas as pd
import numpy as np
import scipy
from scipy.stats import linregress, norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import gridspec

df = pd.read_excel(r"C:\Users\picen\OneDrive\Desktop\CAMPARI\rendimenti_campari.xlsx")
df.columns = ['Data', 'Prezzo']
df = df.iloc[2:]

df['Prezzo'] = pd.to_numeric(df['Prezzo'], errors='coerce')  #Il valore era testo, l'ho trasformato in numero

print(df['Prezzo'].dtype)
df['Rendimenti'] = np.log(df['Prezzo']).diff()
df = df.dropna()  #Per togliere il primo NaN che si ha
print(df.head())

df_start = df.copy()

#Calcolo rendimenti
media_rendimenti_giornalieri = df_start['Rendimenti'].mean()
media_rendimenti_annui = media_rendimenti_giornalieri *250
print(media_rendimenti_annui)

#Calcolo volatilità
sigma_giornaliero = df_start['Rendimenti'].std()
sigma_annuo = sigma_giornaliero*np.sqrt(250)
print(sigma_annuo)

#Ora faccio lo stesso per l'indice FTSE Europe
df_indice = pd.read_excel(r"C:\Users\picen\OneDrive\Desktop\CAMPARI\rendimenti_campari.xlsx", 
                          sheet_name="Indici")

prezzo_benchmark = df_indice.iloc[:,1:].apply(pd.to_numeric)
rendimento_benchmark = np.log(prezzo_benchmark).diff().dropna()

media_rendimento_annuo_benchmark = rendimento_benchmark.mean()*250
sigma_annuo_benchmark = rendimento_benchmark.std()*np.sqrt(250)

#Tasso di Crescita degli Asset
rend_titolo = df['Rendimenti']
print(rend_titolo.shape)
slope, intercept, r_value, p_value, std_err = linregress(rendimento_benchmark.iloc[:,0], rend_titolo)
beta = slope
print("Beta:", beta)

tasso_crescita_asset = (beta*media_rendimento_annuo_benchmark).item()
print(tasso_crescita_asset)

# --- DEFINIZIONE DEL SISTEMA ---
def merton_system(x):
    V0, sigma = x
    D = 3131554.89
    E = 4005269.40
    T = 1
    # Formule d1 e d2
    d1 = (np.log(V0 / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    print(d1,d2)
    
    # Equazione 1: Prezzo dell'Equity (deve corrispondere a E)
    S0_calc = V0 * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    
    # Equazione 2: Relazione tra volatilità Equity (sigma_annuo) e Asset (sigma)
    sigma_S_calc = (norm.cdf(d1) * sigma * V0) / S0_calc
    
    # Restituiamo la differenza tra valori calcolati e valori reali
    return [S0_calc - E, sigma_S_calc - sigma_annuo]

# --- RISOLUZIONE ---
# Punto di partenza: V0 = Somma semplice di E e D, sigma = volatilità equity
start_Camp = [E + D, sigma_annuo] 

# Risolviamo il sistema
sol_Camp = fsolve(merton_system, start_Camp)

# Estraiamo i risultati
V0, sigma = sol_Camp
print(f"Asset Value (V0): {V0:.2f}")
print(f"Asset Volatility (sigma): {sigma:.4%}")

######Calcolo EDF
# Approccio KMV
tasso_crescita_asset  # Il tasso di crescita Mu che abbiamo calcolato con la regressione
debiti_breve =  305748.13 
debiti_ml =  2825806.75 
default_point = debiti_breve + 0.5*debiti_ml
print(default_point)
r = 0.036

# Distanza dal default (DD)
d2_KMV = (np.log(V0 / default_point) + (tasso_crescita_asset - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
EDF_empirica = norm.cdf(-d2_KMV)
print(d2_KMV)
# Probabilità Risk-Neutral
sigma_annuo_benchmark
sharpe = (tasso_crescita_asset - r) / sigma_annuo_benchmark
d2_rn = d2_KMV - sharpe
EDF_risk_neutral = norm.cdf(-d2_rn)

print(f"EDF Empirica: {EDF_empirica.item():.20f}")
print(f"EDF Risk-Neutral: {EDF_risk_neutral.item():.20f}")

#### Simulazioni MonteCarlo
nsim = 30000
months = 12
dt = 1/12
t = np.arange(months + 1)

# Simulazione Geometric Brownian Motion
Z = np.random.standard_normal((nsim, months))
V_paths = np.zeros((nsim, months + 1))
V_paths[:, 0] = V0

for step in range(1, months + 1):
    V_paths[:, step] = V_paths[:, step-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, step-1])

# 1. Troviamo la traiettoria più vicina alla riga rossa (D)
minimi_per_traiettoria = np.min(V_paths, axis=1)
indice_piu_vicina = np.argmin(np.abs(minimi_per_traiettoria - D))
traiettoria_vicina = V_paths[indice_piu_vicina, :]

# --- GRAFICO A SINISTRA: Traiettorie ---
ax0 = plt.subplot(gs[0])

# Sfondo: tutte le 10.000 traiettorie
ax0.plot(t, V_paths.T, color='steelblue', lw=0.1, alpha=0.01)

# Evidenziazione: traiettoria più vicina (midnightblue e sottile)
ax0.plot(t, traiettoria_vicina, color='midnightblue', lw=1.2, label='Closest Path to Default Point')

# Barriera del debito
ax0.axhline(y=D, color='red', linestyle='-', linewidth=2, label='Default Point (D)')

ax0.set_ylim(D * 0.8, V0 * 1.3) # Regolazione limiti per visibilità
ax0.set_title(f"Merton Model: {nsim} Simulations")
ax0.set_xlabel("Months")
ax0.set_ylabel("Asset Value")
ax0.legend(loc='upper left')
plt.show()
plt.savefig("fig.png", dpi=300, bbox_inches="tight")

# Output di verifica
print(f"La traiettoria evidenziata è arrivata a un valore minimo di: {minimi_per_traiettoria[indice_piu_vicina]:.2f}")


print(f"Distanza minima dalla riga rossa: {abs(minimi_per_traiettoria[indice_piu_vicina] - D):.2f}")
