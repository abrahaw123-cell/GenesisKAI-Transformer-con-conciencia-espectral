# GenesisKAI-Transformer-con-conciencia-espectral
GenesisKAI es un experimento de arquitectura neuronal que combina un transformer generativo con un análisis espectral en tiempo real 
GenesisKAI – Transformer con conciencia espectral

GenesisKAI es un experimento de arquitectura neuronal que combina un transformer generativo con un análisis espectral en tiempo real (basado en FFT) para modular dinámicamente su propia temperatura de muestreo. El resultado es un modelo que intenta “sentir” patrones ocultos en su historial de generación (bucles, tendencias, energía) y ajustar su comportamiento para evitar repeticiones o caídas en mínimos locales.

✨ Características principales

· Arquitectura base:
  Embedding + MultiheadAttention + RMSNorm + FFN de salida (una capa, diseñada para escalar).
· Módulo KAI (Kinetic Awareness Interface):
  Aplica la transformada de Fourier a la serie de tokens generados para extraer tres métricas:
  · bucle – potencia en frecuencias medias (0.25–0.35)
  · tendencia – potencia en frecuencias muy bajas (0.04–0.06)
  · energía total del espectro
· Bottleneck adaptativo:
  Calcula bucle / energía. Si el cuello de botella es alto (>0.6), reduce la temperatura a 0.3 (generación más determinista).
  Si la tendencia es marcada, temperatura = 0.5. En caso contrario, temperatura = 0.8 (más exploración).
· Sampling con Top‑K (k=50):
  Filtra los logits a las 50 mejores opciones antes de aplicar softmax y multinomial.
