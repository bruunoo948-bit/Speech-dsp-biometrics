# Procesado Digital de Voz y Biometría Acústica

Implementación en Python de pipelines clásicos de extracción de características acústicas desde cero y modelos de aprendizaje automático para verificación de locutor y clasificación vocal en tiempo real.

## Pipeline de Procesado de Señal
Toda la extracción de parámetros está programada directamente sobre `NumPy` y `SciPy` sin depender de extractores de alto nivel:
* Pre-procesado: Filtro de pre-énfasis (alpha = 0.97) y enventanado de Hamming (25 ms} con salto de 10 ms.
Análisis Espectral y Mel: RFFT, espectro de potencia y banco de 26 filtros triangulares en escala Mel[cite: 2, 4].
* Parametrización Cepstral: DCT-II para obtención de 13 coeficientes MFCC, normalización cepstral de media (CMN) y cálculo de derivadas primera y segunda (Delta y DeltaDelta), formando vectores de 39 características por trama.
* Detección de Actividad Vocal (VAD): Filtrado de silencios basado en umbral de energía log-Mel.

## Modelos e Interfaces
* Verificación de Locutor (Biometría): Clasificador SVM-RBF entrenado con tramas balanceadas para autenticación de identidad[cite: 4].
* Clasificación de Género Vocal: Comparativa entre SVM-RBF y red neuronal MLP (early stopping) con votación por mayoría por trama.
* Demo en Tiempo Real: Interfaz web interactiva desarrollada con Gradio para captura y evaluación directa desde micrófono a 16kHz.

## Requisitos
```bash
pip install numpy scipy scikit-learn librosa soundfile gradio joblib
