# genesis_kai.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────
# MODELO BASE
# ─────────────────────────────────────────────
class GenesisKAI(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.fc(x)

# ─────────────────────────────────────────────
# DFT / KAI ANALYSIS
# ─────────────────────────────────────────────
def compute_kai(series):
    if len(series) < 8:
        return {"bucle": 0, "tendencia": 0, "energia": 0}

    x = np.array(series)
    x = x - np.mean(x)

    fft = np.fft.fft(x)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(len(power))

    energia_total = np.sum(power)

    bucle_mask = (freqs > 0.25) & (freqs < 0.35)
    tend_mask = (freqs > 0.04) & (freqs < 0.06)

    bucle = np.max(power[bucle_mask]) if np.any(bucle_mask) else 0
    tendencia = np.max(power[tend_mask]) if np.any(tend_mask) else 0

    return {
        "bucle": float(bucle),
        "tendencia": float(tendencia),
        "energia": float(energia_total)
    }

def compute_bottleneck(kai):
    if kai["energia"] == 0:
        return 0
    return kai["bucle"] / kai["energia"]

# ─────────────────────────────────────────────
# GENERACIÓN INTELIGENTE
# ─────────────────────────────────────────────
def generar_respuesta_inteligente(modelo, tokens, historial_S):
    modelo.eval()

    with torch.no_grad():
        logits = modelo(tokens)
        logits = logits[:, -1, :]

        # ANALÍTICA SISTÉMICA
        kai = compute_kai(historial_S)
        bottleneck = compute_bottleneck(kai)

        # AJUSTE DINÁMICO
        if bottleneck > 0.6:
            temperature = 0.3
        elif kai["tendencia"] > 0.02:
            temperature = 0.5
        else:
            temperature = 0.8

        logits = logits / temperature

        # TOP-K
        top_k = 50
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token, {
            "bottleneck": float(bottleneck),
            "tendencia": float(kai["tendencia"]),
            "temperature": temperature
        }
