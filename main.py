import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Plot por coluna (1000 Hz)", layout="wide")

st.title("Visualizador de séries temporais por coluna")
st.caption("Carregue um arquivo (.xlsx/.csv). Cada coluna numérica vira um gráfico separado. fs = 1000 Hz.")

FS_DEFAULT = 1200.0

uploaded = st.file_uploader("Escolha o arquivo", type=["xlsx", "xls", "csv"])

# -------------------------
# Helpers
# -------------------------
def _read_file(file, file_name: str):
    if file_name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(file)
        return xls, xls.sheet_names
    else:
        df = pd.read_csv(file)
        return df, None

def _to_numeric_df(df: pd.DataFrame):
    # Converte o que der para número e remove colunas completamente vazias/não-numéricas
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out

def _make_time(n: int, fs: float):
    return np.arange(n, dtype=float) / fs

# -------------------------
# UI
# -------------------------
if uploaded is None:
    st.info("Envie um arquivo para começar.")
    st.stop()

fs = st.number_input("Frequência de amostragem (Hz)", min_value=1.0, value=FS_DEFAULT, step=1.0)

data_obj, sheet_names = _read_file(uploaded, uploaded.name)

if sheet_names is not None:
    col1, col2 = st.columns([2, 3])
    with col1:
        sheet = st.selectbox("Aba (sheet)", sheet_names)
    with col2:
        header_row = st.number_input("Linha do cabeçalho (0 = primeira linha)", min_value=0, value=0, step=1)
    df_raw = pd.read_excel(data_obj, sheet_name=sheet, header=int(header_row))
else:
    df_raw = data_obj

if df_raw is None or df_raw.empty:
    st.error("Não consegui ler dados do arquivo (dataframe vazio).")
    st.stop()

st.subheader("Pré-visualização")
st.dataframe(df_raw.head(20), use_container_width=True)

# Detecta colunas
cols = list(df_raw.columns)
if len(cols) == 0:
    st.error("Não encontrei colunas no arquivo.")
    st.stop()

st.subheader("Configuração do eixo do tempo")

time_mode = st.radio(
    "Como criar o eixo do tempo?",
    ["Gerar pelo fs (1000 Hz)", "Usar uma coluna de tempo existente"],
    horizontal=True,
)

t_col = None
if time_mode == "Usar uma coluna de tempo existente":
    t_col = st.selectbox("Escolha a coluna de tempo", cols)

# Data numérica
if time_mode == "Usar uma coluna de tempo existente":
    t = pd.to_numeric(df_raw[t_col], errors="coerce").to_numpy()
    # Remove linhas onde tempo é NaN
    mask = ~np.isnan(t)
    df_plot_base = df_raw.loc[mask].copy()
    t = t[mask]
    # Converte outras colunas para numérico
    df_num = _to_numeric_df(df_plot_base.drop(columns=[t_col], errors="ignore"))
else:
    df_num = _to_numeric_df(df_raw)
    t = _make_time(len(df_num), fs)

if df_num.empty:
    st.error("Não há colunas numéricas para plotar após conversão.")
    st.stop()

st.subheader("O que plotar")
num_cols = list(df_num.columns)

colA, colB, colC = st.columns([2, 2, 2])
with colA:
    mode = st.radio("Modo", ["Plotar todas as colunas (gráficos separados)", "Selecionar colunas"], horizontal=False)
with colB:
    max_plots = st.number_input("Máx. de gráficos (evita travar)", min_value=1, value=min(30, len(num_cols)), step=1)
with colC:
    downsample = st.number_input("Downsample (pegar 1 a cada N pontos)", min_value=1, value=1, step=1)

if mode == "Selecionar colunas":
    chosen = st.multiselect("Selecione as colunas", num_cols, default=num_cols[:min(5, len(num_cols))])
else:
    chosen = num_cols

# Aplica downsample
idx = np.arange(0, len(t), int(downsample))
t_ds = t[idx]
df_ds = df_num.iloc[idx].copy()

# Limita número de gráficos
chosen = chosen[: int(max_plots)]

st.divider()
st.subheader("Gráficos")

# Usando line_chart (rápido) em gráficos separados
# Para cada coluna, montamos um DataFrame com índice tempo.
for c in chosen:
    series = pd.to_numeric(df_ds[c], errors="coerce")
    if series.isna().all():
        continue

    chart_df = pd.DataFrame({c: series.to_numpy()}, index=t_ds)
    chart_df.index.name = "tempo (s)"

    with st.expander(f"Coluna: {c}", expanded=True):
        st.line_chart(chart_df, height=220, use_container_width=True)

st.success(f"Pronto! Colunas plotadas: {len(chosen)} / {len(num_cols)}")
