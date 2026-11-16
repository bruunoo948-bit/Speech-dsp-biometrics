# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:23:26 2026

@author: bruun
"""

import os
import glob
import numpy as np
import scipy.fftpack
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import time
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
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

FS_TARGET = 16000

def construir_dataset(dir_base="dataset"):
    X, y, locutores = [], [], []

    # Configuración de clases: Bruno (1) vs Otros (0)
    clases = [
        (os.path.join(dir_base, "bruno"), 1),
        (os.path.join(dir_base, "otros"), 0)
    ]

    # Formatos de audio soportados
    extensiones = ('*.wav', '*.flac', '*.ogg', '*.m4a', '*.mp3', '*.opus')

    for carpeta, etiqueta in clases:
        archivos = []
        for ext in extensiones:
            archivos.extend(glob.glob(os.path.join(carpeta, ext)))
            archivos.extend(glob.glob(os.path.join(carpeta, '**', ext), recursive=True))

        for ruta_fichero in archivos:
            nombre_fichero = os.path.basename(ruta_fichero)
            locutor = "bruno" if etiqueta == 1 else nombre_fichero.split('-')[0]

            # Carga el audio (mono) y re-muestrea automáticamente a 16 kHz
            x, fs_audio = librosa.load(ruta_fichero, sr=FS_TARGET, mono=True)

            # Extraer características
            lm, mf = extraer_pipeline(x, fs_audio, duracion_trama, saltos_trama, coeficiente_preenfasis, N_fft)
            
            mf = cmn(mf)
            d1 = calcular_delta(mf)
            d2 = calcular_delta(d1)
            feat = np.vstack([mf, d1, d2])  # Forma: (39, T_tramas)

            # Filtro VAD por energía
            energia = lm.mean(axis=0)
            voz = energia > np.percentile(energia, 30)
            feat_voz = feat[:, voz]  # Forma: (39, T_tramas_voz)

            cant_tramas = feat_voz.shape[1]
            if cant_tramas > 0:
                X.append(feat_voz.T)  # (T_tramas_voz, 39)
                y += [etiqueta] * cant_tramas
                locutores += [locutor] * cant_tramas

    X = np.vstack(X)
    y = np.array(y)
    locutores = np.array(locutores)

    print(f"Dataset listo: {X.shape[0]} tramas x {X.shape[1]} dims "
          f"| Bruno (1): {(y==1).sum()} tramas | Otros (0): {(y==0).sum()} tramas")

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

# Construcción del dataset global
X, y, locutores = construir_dataset()

# CORRECCIÓN 1: Filtrar por los enteros 1 (Bruno) y 0 (Otros)
idx_b = np.where(y == 1)[0]
idx_o = np.where(y == 0)[0]

rng = np.random.default_rng(42)

# ==========================================
# ------- 2. ENTRENAMIENTO SVM-RBF ---------
# ==========================================
N_POR_CLASE_SVM = 15000

idx_b_svm = rng.choice(idx_b, size=min(N_POR_CLASE_SVM, len(idx_b)), replace=False)
idx_o_svm = rng.choice(idx_o, size=min(N_POR_CLASE_SVM, len(idx_o)), replace=False)

idx_bal_svm = np.hstack([idx_b_svm, idx_o_svm])
rng.shuffle(idx_bal_svm)

X_bal_svm = X[idx_bal_svm]
y_bal_svm = y[idx_bal_svm]

# --- LÍNEA AÑADIDA: Divisió Train (80%) / Test (20%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_bal_svm, y_bal_svm, test_size=0.20, random_state=42, stratify=y_bal_svm
)

print(f"\nDataset balanceado total: {(y_bal_svm == 1).sum()} Bruno (1) vs {(y_bal_svm == 0).sum()} Otros (0)")
print(f"Conjunto de Train: {len(X_train)} tramas | Conjunto de Test: {len(X_test)} tramas")

# Escalar SOLO con datos de entrenamiento
esc_svm = StandardScaler()
X_n_svm = esc_svm.fit_transform(X_train)

# Instanciar y entrenar SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')

tic()
svm.fit(X_n_svm, y_train)
t_fit_svm = toc("Entrenamiento SVM")

# Guardar SVM y su escalador
joblib.dump(svm, "modelo_svm_locutor.pkl")
joblib.dump(esc_svm, "escalador_std.pkl")


# ==========================================
# ------- 4. EVALUACIÓN SVM-RBF ------------
# ==========================================
# Transformar Test usando el escalador de Train
X_test_svm = esc_svm.transform(X_test)

tic()
acc_tr_svm = svm.score(X_n_svm, y_train)
acc_te_svm = svm.score(X_test_svm, y_test)
pred_svm = svm.predict(X_test_svm)
t_eval_svm = toc("Evaluación e Inferencia SVM")

gap_svm = (acc_tr_svm - acc_te_svm) * 100

# CORRECCIÓN: Labels numéricos [0, 1] en lugar de ['F', 'M']
cm_svm = confusion_matrix(y_test, pred_svm, labels=[0, 1])

print("\n--- EVALUACIÓN SVM-RBF ---")
print(f"Accuracy Train: {acc_tr_svm * 100:.2f}%")
print(f"Accuracy Test:  {acc_te_svm * 100:.2f}%")
print(f"Brecha (Gap):    {gap_svm:.2f} pts")
print("\nMatriz de confusión (Test) [Filas: Real, Col: Pred] -> [Otros (0), Bruno (1)]:\n", cm_svm)
print("\nInforme de Clasificación Detallado:\n", classification_report(y_test, pred_svm, target_names=['Otros', 'Bruno']))
print("-" * 50)

