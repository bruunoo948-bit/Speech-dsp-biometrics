import gradio as gr
import numpy as np
import soundfile as sf
import joblib



import os
import glob
import numpy as np
import scipy.fftpack
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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



# Cargar el modelo entrenado y el escalador guardados
svm = joblib.load("modelo_svm_genero.pkl")
esc = joblib.load("escalador_std.pkl")

# Parámetros del pipeline (idénticos a los del entrenamiento)
FS_TARGET = 16000  # Frecuencia exacta con la que se entrenó el dataset
duracion_trama = 0.025
saltos_trama = 0.010
coeficiente_preenfasis = 0.97
N_fft = 512

def predecir_genero_microfono(audio_path):
    if audio_path is None:
        return "Por favor, graba un audio para clasificar."

    #Cargar la señal grabada desde la web
    x, fs_audio = sf.read(audio_path)
    if x.ndim > 1:
        x = x.mean(axis=1)

    #RE-MUESTREO OBLIGATORIO A 16.000 Hz
    if fs_audio != FS_TARGET:
        num_muestras_nuevas = int(round(len(x) * float(FS_TARGET) / fs_audio))
        x = scipy.signal.resample(x, num_muestras_nuevas)
        fs_audio = FS_TARGET

    # Normalizar amplitud del audio del micro para evitar audios muy flojos
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))

    #Aplicar pipeline de extracción de características
    lm, mf = extraer_pipeline(x, fs_audio, duracion_trama, saltos_trama, coeficiente_preenfasis, N_fft)
    mf = cmn(mf)
    d1 = calcular_delta(mf)
    d2 = calcular_delta(d1)
    feat = np.vstack([mf, d1, d2])  # Dimensión (39, T)

    #Filtrar tramas de silencio por energía
    energia = lm.mean(axis=0)
    voz = energia > np.percentile(energia, 30)
    feat_voz = feat[:, voz].T  # Dimensión (T_voz, 39)

    if len(feat_voz) == 0:
        return "No se ha detectado suficiente voz en la grabación."

    #Normalizar tramas con el MISMO escalador del entrenamiento
    feat_voz_norm = esc.transform(feat_voz)

    #Predicción trama a trama y votación por mayoría
    predicciones = svm.predict(feat_voz_norm)
    votos_m = np.sum(predicciones == 'M')
    votos_f = np.sum(predicciones == 'F')
    total_tramas = len(predicciones)

    #Generar resultado
    if votos_m >= votos_f:
        confianza = (votos_m / total_tramas) * 100
        return f"Voz Masculina (Confianza: {confianza:.1f}% | {votos_m}/{total_tramas} tramas)"
    else:
        confianza = (votos_f / total_tramas) * 100
        return f"Voz Femenina (Confianza: {confianza:.1f}% | {votos_f}/{total_tramas} tramas)"

#Montar e iniciar la interfaz gráfica
demo = gr.Interface(
    fn=predecir_genero_microfono,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="Clasificador de Género por Voz",
    description="Pulsa el botón del micrófono, habla durante 2-4 segundos y pulsa **Submit**."
)

demo.launch()

demo.launch(inbrowser=True)