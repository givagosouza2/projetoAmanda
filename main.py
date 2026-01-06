import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Configuração da página
# ======================================================
st.set_page_config(
    page_title="Layout 2x3 – Séries Temporais",
    layout="wide"
)

st.title("Visualização 2 × 3 de séries temporais (Matplotlib)")
st.caption("fs = 1000 Hz | Organização fixa por cabeçalho")

FS_DEFAULT = 1000.0

# ======================================================
# Upload do arquivo
# ======================================================
uploaded_file = st.file_uploader(
    "Carregue o arquivo (.xlsx, .xls ou .csv)",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info("⬆️ Envie um arquivo para iniciar.")
    st.stop()

# ======================================================
# Frequência de amostragem
# ======================================================
fs = st.number_input(
    "Frequência de amostragem (Hz)",
    min_value=1.0,
    value=FS_DEFAULT,
    step=1.0
)

# ======================================================
# Leitura do arquivo
# ======================================================
def read_file(file):
    name = file.name.lower()

    if name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(file)
        except ImportError:
            st.error(
                "Dependência ausente: openpyxl\n\n"
                "Instale com:\n`pip install openpyxl`"
            )
            st.stop()

        sheet = st.selectbox("Selecione a aba (sheet)", xls.sheet_names)
        header_row = st.number_input(
            "Linha do cabeçalho (0 = primeira linha)",
            min_value=0,
            value=0,
            step=1
        )

        df = pd.read_excel(
            xls,
            sheet_name=sheet,
            header=int(header_row)
        )
    else:
        df = pd.read_csv(file)

    return df


df = read_file(uploaded_file)

if df.empty:
    st.error("Arquivo sem dados.")
    st.stop()

# ======================================================
# Converter colunas para numérico
# ======================================================
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.dropna(axis=1, how="all", inplace=True)

# ======================================================
# Tempo
# ======================================================
t = np.arange(len(df)) / fs

# ======================================================
# Ordem fixa desejada
# ======================================================
layout_cols = [
    ["0.4 cpd BW", "2 cpd BW", "6 cpd BW"],  # esquerda
    ["0.4 cpd RG", "2 cpd RG", "6 cpd RG"],  # direita
]

# ======================================================
# Plotagem 2 × 3
# ======================================================
st.divider()
st.subheader("Gráficos organizados (2 colunas × 3 linhas)")

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(14, 9),
    sharex=True
)

for col_idx, col_group in enumerate(layout_cols):
    for row_idx, col_name in enumerate(col_group):
        ax = axes[row_idx, col_idx]

        if col_name in df.columns:
            y = df[col_name].values
            ax.plot(t, y, linewidth=0.8)
            ax.set_ylabel(col_name)
        else:
            ax.text(
                0.5,
                0.5,
                f"Coluna ausente:\n{col_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red"
            )

        ax.grid(True, alpha=0.3)

# Rótulos finais
axes[-1, 0].set_xlabel("Tempo (s)")
axes[-1, 1].set_xlabel("Tempo (s)")

axes[0, 0].set_title("BW")
axes[0, 1].set_title("RG")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ======================================================
# Final
# ======================================================
st.success("Layout 2 × 3 gerado com sucesso.")
