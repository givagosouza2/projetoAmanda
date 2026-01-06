import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Configuração da página
# ======================================================
st.set_page_config(
    page_title="Visualizador de Séries Temporais (Matplotlib)",
    layout="wide"
)

st.title("Visualizador de séries temporais por coluna (Matplotlib)")
st.caption("Cada coluna numérica é exibida em um gráfico separado. fs = 1000 Hz.")

FS_DEFAULT = 1000.0

# ======================================================
# Upload
# ======================================================
uploaded_file = st.file_uploader(
    "Carregue um arquivo (.xlsx, .xls ou .csv)",
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
                "Dependência ausente: **openpyxl**\n\n"
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


df_raw = read_file(uploaded_file)

if df_raw.empty:
    st.error("Arquivo carregado, mas sem dados.")
    st.stop()

# ======================================================
# Pré-visualização
# ======================================================
st.subheader("Pré-visualização")
st.dataframe(df_raw.head(20), use_container_width=True)

# ======================================================
# Configuração do tempo
# ======================================================
st.subheader("Configuração do eixo temporal")

time_mode = st.radio(
    "Definição do tempo",
    ["Gerar automaticamente (fs = 1000 Hz)", "Usar coluna existente"],
    horizontal=True
)

cols = list(df_raw.columns)

if time_mode == "Usar coluna existente":
    time_col = st.selectbox("Selecione a coluna de tempo", cols)
    t = pd.to_numeric(df_raw[time_col], errors="coerce").values
    valid = ~np.isnan(t)
    df_work = df_raw.loc[valid].copy()
    t = t[valid]
    df_work.drop(columns=[time_col], inplace=True)
else:
    df_work = df_raw.copy()
    t = np.arange(len(df_work)) / fs

# ======================================================
# Converter colunas para numérico
# ======================================================
for c in df_work.columns:
    df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

df_work.dropna(axis=1, how="all", inplace=True)

if df_work.empty:
    st.error("Nenhuma coluna numérica válida encontrada.")
    st.stop()

numeric_cols = list(df_work.columns)

# ======================================================
# Opções de plotagem
# ======================================================
st.subheader("Opções de visualização")

c1, c2, c3 = st.columns(3)

with c1:
    mode = st.radio(
        "Modo",
        ["Todas as colunas", "Selecionar colunas"]
    )

with c2:
    max_plots = st.number_input(
        "Número máximo de gráficos",
        min_value=1,
        max_value=len(numeric_cols),
        value=min(20, len(numeric_cols)),
        step=1
    )

with c3:
    downsample = st.number_input(
        "Downsample (1 = sem redução)",
        min_value=1,
        value=1,
        step=1
    )

if mode == "Selecionar colunas":
    selected_cols = st.multiselect(
        "Selecione as colunas",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
else:
    selected_cols = numeric_cols

selected_cols = selected_cols[: int(max_plots)]

# ======================================================
# Downsample
# ======================================================
idx = np.arange(0, len(t), int(downsample))
t_ds = t[idx]
df_ds = df_work.iloc[idx].copy()

# ======================================================
# Plotagem com Matplotlib
# ======================================================
st.divider()
st.subheader("Gráficos")

for col in selected_cols:
    y = df_ds[col].values

    if np.all(np.isnan(y)):
        continue

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ds, y, linewidth=0.8)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(col)
    ax.set_title(col)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ======================================================
# Final
# ======================================================
st.success(
    f"{len(selected_cols)} gráficos exibidos "
    f"(de {len(numeric_cols)} colunas numéricas)."
)
