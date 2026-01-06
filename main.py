import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Config
# ======================================================
st.set_page_config(page_title="Cursor global 2x3", layout="wide")
st.title("Séries temporais 2×3 com cursor global (Matplotlib)")
st.caption("fs = 1000 Hz | Slider controla um cursor temporal comum em todos os gráficos.")

FS_DEFAULT = 1200.0

# Ordem fixa desejada
LAYOUT = [
    ["0.4 cpd BW", "2 cpd BW", "6 cpd BW"],  # esquerda
    ["0.4 cpd RG", "2 cpd RG", "6 cpd RG"],  # direita
]

# ======================================================
# Upload
# ======================================================
uploaded_file = st.file_uploader("Carregue o arquivo (.xlsx, .xls ou .csv)", type=["xlsx", "xls", "csv"])
if uploaded_file is None:
    st.info("⬆️ Envie um arquivo para iniciar.")
    st.stop()

fs = st.number_input("Frequência de amostragem (Hz)", min_value=1.0, value=FS_DEFAULT, step=1.0)

def read_file(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(file)
        except ImportError:
            st.error("Dependência ausente: **openpyxl**. Instale com: `pip install openpyxl`")
            st.stop()

        sheet = st.selectbox("Selecione a aba (sheet)", xls.sheet_names)
        header_row = st.number_input("Linha do cabeçalho (0 = primeira linha)", min_value=0, value=0, step=1)
        df = pd.read_excel(xls, sheet_name=sheet, header=int(header_row))
    else:
        df = pd.read_csv(file)
    return df

df_raw = read_file(uploaded_file)
if df_raw.empty:
    st.error("Arquivo sem dados.")
    st.stop()

# Preview
#st.subheader("Pré-visualização")
#st.dataframe(df_raw.head(15), use_container_width=True)

# ======================================================
# Converte para numérico
# ======================================================
df = df_raw.copy()
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.dropna(axis=1, how="all", inplace=True)

n = len(df)
if n < 2:
    st.error("Poucas amostras.")
    st.stop()

t = np.arange(n) / fs
t_end = float(t[-1])

# ======================================================
# Controles do cursor
# ======================================================
st.subheader("Cursor global (tempo e leitura de amplitude)")

colA, colB, colC = st.columns([2, 1, 1])

with colA:
    t0 = st.slider("Tempo do cursor (s)", min_value=0.0, max_value=t_end, value=min(1.0, t_end), step=1.0/fs)
with colB:
    show_zoom = st.checkbox("Zoom ao redor do cursor", value=True)
with colC:
    win = st.number_input("Janela do zoom (s)", min_value=0.01, value=2.0, step=0.25)

# índice da amostra mais próxima
i0 = int(np.clip(np.round(t0 * fs), 0, n - 1))
t0_eff = float(t[i0])

# limites do zoom
if show_zoom:
    half = win / 2.0
    xmin = max(0.0, t0_eff - half)
    xmax = min(t_end, t0_eff + half)
else:
    xmin, xmax = 0.0, t_end

# ======================================================
# Plot 2×3 com linha vertical do cursor
# ======================================================
st.divider()
st.subheader("Gráficos com cursor sincronizado")

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 9), sharex=True)

for col_idx, group in enumerate(LAYOUT):  # 0=BW, 1=RG
    for row_idx, col_name in enumerate(group):
        ax = axes[row_idx, col_idx]

        if col_name in df.columns:
            y = df[col_name].to_numpy()

            # plota a série
            ax.plot(t, y, linewidth=0.8)

            # cursor (linha vertical)
            ax.axvline(t0_eff, linewidth=1.0)

            # marca o ponto no cursor
            if 0 <= i0 < len(y) and not np.isnan(y[i0]):
                ax.plot([t0_eff], [y[i0]], marker="o", markersize=4)

            ax.set_ylabel(col_name)
        else:
            ax.text(
                0.5, 0.5,
                f"Coluna ausente:\n{col_name}",
                ha="center", va="center",
                transform=ax.transAxes
            )

        ax.grid(True, alpha=0.3)
        ax.set_xlim(xmin, xmax)

axes[0, 0].set_title("BW")
axes[0, 1].set_title("RG")
axes[-1, 0].set_xlabel("Tempo (s)")
axes[-1, 1].set_xlabel("Tempo (s)")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.success(f"Cursor em t = {t0_eff:.4f} s (amostra {i0}).")

# ======================================================
# Monta tabela de leituras no cursor
# ======================================================
readouts = []
missing = []

for side in range(2):
    for name in LAYOUT[side]:
        if name in df.columns:
            y = df[name].to_numpy()
            val = float(y[i0]) if not np.isnan(y[i0]) else np.nan
            readouts.append({"Sinal": name, "Tempo (s)": t0_eff, "Amostra": i0, "Amplitude": val})
        else:
            missing.append(name)

st.write("**Leituras no cursor (amostra mais próxima):**")
st.dataframe(pd.DataFrame(readouts), use_container_width=True)

if missing:
    st.warning("Colunas ausentes no arquivo: " + ", ".join(missing))
