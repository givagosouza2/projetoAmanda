# app.py
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plot de séries temporais (Veris .txt)", layout="wide")

st.title("Plot de séries temporais de arquivo .txt (export Veris / tabulado)")

st.markdown(
    """
Este app lê arquivos `.txt` onde:
- Linhas de comentário começam com `%`
- Cada **linha de dados** começa com um número (ex.: `1`) e contém valores separados por TAB
- A 1ª coluna é o **grupo**, e as demais colunas são as amostras da **trace**
"""
)

def _decode_bytes(data: bytes) -> str:
    # tenta UTF-8; se falhar, cai para latin-1 (comum em alguns exports)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")

_num_re = re.compile(r"^[\s]*[-+]?\d")  # linha começando com dígito/sinal (dado)

def parse_veris_tabbed(text: str) -> pd.DataFrame:
    """
    Retorna DataFrame no formato wide:
    - group: id do grupo
    - trace_000, trace_001, ...: colunas com a série temporal (mesmo comprimento, com NaN se variar)
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            continue
        # pula header tipo "Group number   Traces"
        if line.lower().startswith("group number"):
            continue
        if not _num_re.match(line):
            continue

        parts = re.split(r"\t+", line)
        if len(parts) < 3:
            continue

        try:
            group = int(float(parts[0].strip()))
        except Exception:
            continue

        vals = []
        ok = True
        for p in parts[1:]:
            p = p.strip()
            if p == "":
                continue
            try:
                vals.append(float(p))
            except Exception:
                ok = False
                break
        if ok and len(vals) > 0:
            rows.append((group, vals))

    if not rows:
        raise ValueError("Nenhuma linha de dados numéricos válida foi encontrada.")

    max_len = max(len(v) for _, v in rows)
    data = {"group": [g for g, _ in rows]}
    for i, (_, v) in enumerate(rows):
        col = f"trace_{i:03d}"
        arr = np.full(max_len, np.nan, dtype=float)
        arr[: len(v)] = np.asarray(v, dtype=float)
        data[col] = arr

    return pd.DataFrame(data)

uploaded = st.file_uploader("Carregue o arquivo .txt", type=["txt"])

if uploaded is not None:
    text = _decode_bytes(uploaded.read())

    with st.expander("Ver prévia do arquivo (primeiras linhas)"):
        st.code("\n".join(text.splitlines()[:40]))

    try:
        df = parse_veris_tabbed(text)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    n_traces = df.shape[1] - 1
    n_samples = df.shape[0]
    st.success(f"OK! Encontrei {n_traces} trace(s) com {n_samples} amostras (NaNs possíveis no fim).")

    # ---------- Controles ----------
    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        available = [c for c in df.columns if c.startswith("trace_")]
        selection = st.multiselect(
            "Escolha as traces para plotar",
            options=available,
            default=available[:1] if available else [],
        )
    with colB:
        y_mode = st.selectbox("Eixo Y", ["Volts (V)", "Microvolts (µV)"])
        detrend = st.checkbox("Remover tendência (detrend linear)", value=False)
    with colC:
        x_mode = st.selectbox("Eixo X", ["Índice da amostra", "Tempo (s)"])
        fs = None
        dt = None
        if x_mode == "Tempo (s)":
            mode = st.radio("Definir tempo por", ["Frequência (Hz)", "Δt (s)"], horizontal=True)
            if mode == "Frequência (Hz)":
                fs = st.number_input("Frequência de amostragem (Hz)", min_value=0.0001, value=1000.0, step=1.0)
                dt = 1.0 / float(fs)
            else:
                dt = st.number_input("Δt (s)", min_value=0.0, value=0.001, step=0.0001, format="%.6f")

    if not selection:
        st.warning("Selecione pelo menos uma trace para plotar.")
    else:
        # ---------- Monta eixo X ----------
        if x_mode == "Tempo (s)" and dt is not None and dt > 0:
            x = np.arange(n_samples) * float(dt)
            xlabel = "Tempo (s)"
        else:
            x = np.arange(n_samples)
            xlabel = "Amostra (índice)"

        # ---------- Plot ----------
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for c in selection:
            y = df[c].to_numpy(dtype=float)

            # detrend simples (linear) sem scipy
            if detrend:
                msk = np.isfinite(y)
                if msk.sum() >= 2:
                    p = np.polyfit(x[msk], y[msk], deg=1)
                    y = y - (p[0] * x + p[1])

            if y_mode == "Microvolts (µV)":
                y = y * 1e6
                ylabel = "Amplitude (µV)"
            else:
                ylabel = "Amplitude (V)"

            ax.plot(x, y, label=c)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, clear_figure=True)

    # ---------- Export ----------
    st.subheader("Exportar")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV (todas as traces)",
        data=csv,
        file_name="traces_export.csv",
        mime="text/csv",
    )

else:
    st.info("Faça upload de um `.txt` para começar.")
