"""
Proyecto personal. Clasificador de voz en masculina o femenina. 
"""


import os
import glob
import numpy as np
import scipy.fftpack
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
# ------ FUNCIONES DE PROCESAMIENTO DE SEÑAL ------

def hz_a_mel(f):
    return 2595*np.log10(1.0 + f /700.0)

def mel_a_hz(m):
    return 700.0*(10.0**(m/2595.0)-1.0)

def banco_mel(fs,nfft,n_mels=26,fmin=0.0,fmax=None):
    if fmax is None:
        fmax = fs / 2.0
    mels = np.linspace(hz_a_mel(fmin), hz_a_mel(fmax), n_mels + 2)
    hz = mel_a_hz(mels)
    bins = np.floor((nfft + 1) * hz / fs).astype(int)
    # Filtros triangulares
    H = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        izq, cen, dch = bins[m-1], bins[m], bins[m+1]
        if cen > izq:
            H[m-1, izq:cen] = (np.arange(izq, cen)-izq) / (cen-izq)
        if dch > cen:
            H[m-1, cen:dch] = (dch-np.arange(cen, dch)) / (dch-cen)
    return H

def filtro_preenfasis(x,coeficiente_preenfasis):
    x_pre_enfasis = np.append(x[0], x[1:] - coeficiente_preenfasis * x[:-1]) #Append crea un vector, si no incluyesemos el x[0] nos faltaria una muestra
    return x_pre_enfasis

def ventana(duracion_trama,saltos_trama,fs,x_pre_enfasis):
    # Es importante convertirlos a enteros
    L = int(round(duracion_trama * fs))#Dividimos la señal en tramas de 25ms con saltos de 10ms
    S = int(round(saltos_trama * fs))
    N_datos = len(x_pre_enfasis)

    numero_tramas = int(1 + np.floor((N_datos - L) / S))
    w = np.hamming(L) #Hacerlo con una ventana triangular tiene mucha fuga espectral

    tramas = np.zeros((numero_tramas, L))
    for i in range(numero_tramas):
        inicio = i*S
        fin = inicio + L
        tramas[i, :] = x_pre_enfasis[inicio:fin] * w
    
    return tramas

def transformada_coseno(et):
    et_db = 10*np.log10(et+1e-10)
    et_transformada_DCT = scipy.fftpack.dct(et_db, type=2, norm='ortho', axis=-1)
    mfcc_13 = et_transformada_DCT[:, :13] #Nos quedamos con los 13 primeros
    return mfcc_13, et_db

def extraer_pipeline(x, fs, duracion_trama, saltos_trama, coeficiente_preenfasis, N_fft):
    

    x_pre_enfasis = filtro_preenfasis(x, coeficiente_preenfasis)
    tramas = ventana(duracion_trama, saltos_trama, fs, x_pre_enfasis)
    espectro = np.fft.rfft(tramas,n = N_fft)
    espectro_potencia = (np.abs(espectro) ** 2)/N_fft
    H = banco_mel(fs,N_fft,n_mels=26,fmin=0.0,fmax=None)
    et = espectro_potencia @ H.T
    mfcc_13, et_db = transformada_coseno(et)
    mfcc = mfcc_13.T                     
    log_mel = et_db.T
    return log_mel, mfcc

def calcular_delta(feat,N=2):
    
    num_coef,T = feat.shape
    dt = np.zeros((num_coef, T))
    pad = np.pad(feat, ((0, 0), (N, N)), mode='edge')
    denominador = 10
    
    for t in range(T):
        numerador = 0
        for n in range(1,N+1):
            numerador += n*(pad[:,t+n+N]-pad[:,t+N-n])
        dt[:,t] = numerador / denominador
    return dt    

def cmn(mfcc):
    media = np.mean(mfcc, axis=1, keepdims=True) #Axis para que se recorra en el tiempo. Keepdims forma una columna 
    mfcc_normalizado_media = mfcc - media
    return mfcc_normalizado_media

def construir_dataset():
    X, y, locutores = [], [], []

    carpetas = [
        (DIR_HOMBRE, 'M'),
        (DIR_MUJER, 'F')
    ]

    for carpeta, genero in carpetas:
        archivos = (glob.glob(os.path.join(carpeta, '*.flac')) + 
                    glob.glob(os.path.join(carpeta, '*.wav')))

        for ruta_fichero in archivos:
            nombre_fichero = os.path.basename(ruta_fichero)
            locutor = nombre_fichero.split('-')[0]

            # Leer archivo de audio
            x, fs_audio = sf.read(ruta_fichero)

            # Convertir estéreo a mono si aplica
            if x.ndim > 1:
                x = x.mean(axis=1)

            # Extraer log_mel y mfcc (2 elementos)
            lm, mf = extraer_pipeline(x, fs_audio, duracion_trama, saltos_trama, coeficiente_preenfasis, N_fft)
            
            mf = cmn(mf)
            d1 = calcular_delta(mf)
            d2 = calcular_delta(d1)
            feat = np.vstack([mf, d1, d2])  # Vector final de 39 características por trama (13 + 13 + 13)

            # Selección de tramas con voz por energía en log_mel
            energia = lm.mean(axis=0)
            voz = energia > np.percentile(energia, 30)

            cant_tramas_voz = int(voz.sum())
            if cant_tramas_voz > 0:
                X.append(feat[:, voz].T)     # Matriz de forma (T_voz, 39)
                y += [genero] * cant_tramas_voz
                locutores += [locutor] * cant_tramas_voz

    X = np.vstack(X)
    y = np.array(y)
    locutores = np.array(locutores)

    print(f"Dataset: {X.shape[0]} tramas x {X.shape[1]} dims "
          f"(F: {(y=='F').sum()}, M: {(y=='M').sum()})")

    return X, y, locutores

_tic_start_time = None

def tic():
    """Inicia el cronómetro (equivalente a tic en MATLAB)."""
    global _tic_start_time
    _tic_start_time = time.perf_counter()

def toc(etiqueta="Tiempo transcurrido"):
    """Mide el tiempo transcurrido desde el último tic() (equivalente a toc en MATLAB)."""
    if _tic_start_time is None:
        print("Error: Debes llamar primero a tic()")
        return 0.0
    tiempo = time.perf_counter() - _tic_start_time
    print(f"⏱️ {etiqueta}: {tiempo:.2f} segundos")
    return tiempo

# ------ 1. PARÁMETROS Y EXTRACCIÓN ------
duracion_trama = 0.025
saltos_trama = 0.010
coeficiente_preenfasis = 0.97
N_fft = 512

DIR_HOMBRE = 'dataset/Hombre'
DIR_MUJER = 'dataset/Mujer'

# Construcción dataset global
X, y, locutores = construir_dataset()

# Índices globales por género
idx_m = np.where(y == 'M')[0]
idx_f = np.where(y == 'F')[0]

rng = np.random.default_rng(42)


# ==========================================
# ------- 2. ENTRENAMIENTO SVM-RBF ---------
# ==========================================
N_POR_CLASE_SVM = 15000

idx_m_svm = rng.choice(idx_m, size=min(N_POR_CLASE_SVM, len(idx_m)), replace=False)
idx_f_svm = rng.choice(idx_f, size=min(N_POR_CLASE_SVM, len(idx_f)), replace=False)

idx_bal_svm = np.hstack([idx_m_svm, idx_f_svm])
rng.shuffle(idx_bal_svm)

X_bal_svm = X[idx_bal_svm]
y_bal_svm = y[idx_bal_svm]

print(f"\nEntrenando SVM con dataset balanceado: {(y_bal_svm=='M').sum()} M vs {(y_bal_svm=='F').sum()} F")

# Escalar y entrenar SVM
esc_svm = StandardScaler()
X_n_svm = esc_svm.fit_transform(X_bal_svm)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')

tic()
svm.fit(X_n_svm, y_bal_svm)
t_fit_svm = toc("Entrenamiento SVM")

# Guardar SVM y su escalador
joblib.dump(svm, "modelo_svm_genero.pkl")
joblib.dump(esc_svm, "escalador_std.pkl")


# ==========================================
# --------- 3. ENTRENAMIENTO MLP -----------
# ==========================================
N_POR_CLASE_MLP = 105000

idx_m_mlp = rng.choice(idx_m, size=min(N_POR_CLASE_MLP, len(idx_m)), replace=False)
idx_f_mlp = rng.choice(idx_f, size=min(N_POR_CLASE_MLP, len(idx_f)), replace=False)

idx_bal_mlp = np.hstack([idx_m_mlp, idx_f_mlp])
rng.shuffle(idx_bal_mlp)

X_bal_mlp = X[idx_bal_mlp]
y_bal_mlp = y[idx_bal_mlp]

# Escalar los datos para MLP
esc_mlp = StandardScaler()
X_n_mlp = esc_mlp.fit_transform(X_bal_mlp)

# Convertir etiquetas a numérico (1 = M, 0 = F)
y_num_mlp = np.where(y_bal_mlp == 'M', 1, 0)

print(f"\nEntrenando MLP con dataset balanceado: {(y_bal_mlp=='M').sum()} M vs {(y_bal_mlp=='F').sum()} F")

# Red con Early Stopping
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=0
)

tic()
mlp.fit(X_n_mlp, y_num_mlp)
t_fit_mlp = toc("Entrenamiento MLP")

# Guardar MLP y su escalador
joblib.dump(mlp, "modelo_mlp_genero.pkl")
joblib.dump(esc_mlp, "escalador_mlp_std.pkl")


# ==========================================
# ----- 4. CONSTRUCCIÓN DATASET TEST -------
# ==========================================
DIR_HOMBRE = 'dataset/Test/Hombre'
DIR_MUJER = 'dataset/Test/Mujer'

X_test, y_test, locutores_test = construir_dataset()
y_test_num = np.where(y_test == 'M', 1, 0)


# ==========================================
# ------- 5. EVALUACIÓN SVM-RBF ------------
# ==========================================
X_test_svm = esc_svm.transform(X_test)

tic()
acc_tr_svm = svm.score(X_n_svm, y_bal_svm)
acc_te_svm = svm.score(X_test_svm, y_test)
pred_svm = svm.predict(X_test_svm)
t_eval_svm = toc("Evaluación e Inferencia SVM")

gap_svm = (acc_tr_svm - acc_te_svm) * 100
cm_svm = confusion_matrix(y_test, pred_svm, labels=['F', 'M'])

print("\n--- SVM-RBF ---")
print(f"Accuracy Train: {acc_tr_svm * 100:.2f}%")
print(f"Accuracy Test:  {acc_te_svm * 100:.2f}%")
print(f"Brecha (Gap):    {gap_svm:.2f} pts")
print("Matriz de confusión (Test) [F, M]:\n", cm_svm)
print("-" * 30)

# ==========================================
# ----------- 6. EVALUACIÓN MLP ------------
# ==========================================
X_test_mlp = esc_mlp.transform(X_test)

tic()
acc_tr_mlp = mlp.score(X_n_mlp, y_num_mlp)
acc_te_mlp = mlp.score(X_test_mlp, y_test_num)
pred_mlp_num = mlp.predict(X_test_mlp)
t_eval_mlp = toc("Evaluación e Inferencia MLP")

gap_mlp = (acc_tr_mlp - acc_te_mlp) * 100
pred_mlp_text = np.where(pred_mlp_num == 1, 'M', 'F')
cm_mlp = confusion_matrix(y_test, pred_mlp_text, labels=['F', 'M'])

print("\n--- MLP ---")
print(f"Accuracy Train: {acc_tr_mlp * 100:.2f}%")
print(f"Accuracy Test:  {acc_te_mlp * 100:.2f}%")
print(f"Brecha (Gap):    {gap_mlp:.2f} pts")
print("Matriz de confusión (Test) [F, M]:\n", cm_mlp)
print("-" * 30)
