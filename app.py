import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Clustering provinsi dan perbandingan Regresi Linear vs Random Forest")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Covid-19_Indonesia_Dataset.csv")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    return df

df = load_data()

# ===============================
# PILIH TANGGAL
# ===============================
tanggal = st.date_input(
    "📅 Pilih Tanggal",
    value=pd.to_datetime("2022-09-15"),
    min_value=df["Tanggal"].min(),
    max_value=df["Tanggal"].max()
)

df_cluster = df[df["Tanggal"] == pd.to_datetime(tanggal)].copy()
df_cluster = df_cluster[df_cluster["Provinsi"].notnull()]

if df_cluster.empty:
    st.error("❌ Data pada tanggal ini tidak tersedia")
    st.stop()

# ===============================
# PILIH FITUR
# ===============================
df_cluster = df_cluster[
    [
        "Provinsi",
        "Total_Kasus",
        "Total_Kematian",
        "Total_Sembuh",
        "Populasi",
        "Kepadatan_Penduduk",
        "Total_Kasus_Per_Juta",
        "Total_Kematian_Per_Juta",
    ]
]

# Rasio
df_cluster["Rasio_Kematian"] = df_cluster["Total_Kematian"] / df_cluster["Total_Kasus"]
df_cluster["Rasio_Kesembuhan"] = df_cluster["Total_Sembuh"] / df_cluster["Total_Kasus"]

df_cluster.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cluster.fillna(0, inplace=True)

fitur = [
    "Total_Kasus",
    "Total_Kematian",
    "Total_Sembuh",
    "Populasi",
    "Kepadatan_Penduduk",
    "Total_Kasus_Per_Juta",
    "Total_Kematian_Per_Juta",
    "Rasio_Kematian",
    "Rasio_Kesembuhan",
]

# ===============================
# MATRIX (CORRELATION MATRIX)
# ===============================
st.subheader("🧮 Matriks Korelasi Fitur")

corr = df_cluster[fitur].corr()

fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
ax_corr.set_title("Correlation Matrix")
st.pyplot(fig_corr)

st.info(
    "📌 Matriks ini menunjukkan hubungan antar variabel. "
    "Nilai mendekati 1 berarti hubungan kuat, mendekati 0 berarti lemah."
)

# ===============================
# SCALING
# ===============================
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df_cluster[fitur])

# ===============================
# CLUSTERING
# ===============================
st.subheader("🔹 Clustering Provinsi (K-Means)")

k = st.slider("Jumlah Cluster (k)", 2, 7, 3)

kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df_cluster["Cluster"] = kmeans.fit_predict(scaled_df)

fig_c, ax_c = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df_cluster,
    x="Kepadatan_Penduduk",
    y="Total_Kematian_Per_Juta",
    hue="Cluster",
    palette="viridis",
    s=100,
    ax=ax_c,
)
ax_c.set_title("Sebaran Cluster Provinsi")
ax_c.grid(True)
st.pyplot(fig_c)

# ===============================
# DATA REGRESI
# ===============================
X = df_cluster[
    ["Populasi", "Kepadatan_Penduduk", "Total_Kasus_Per_Juta"]
]
y = df_cluster["Total_Kematian_Per_Juta"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# REGRESI LINEAR
# ===============================
st.subheader("📈 Regresi Linear")

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_lr = linreg.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

st.write(f"**R² Linear:** {r2_lr:.3f}")
st.write(f"**RMSE Linear:** {rmse_lr:.3f}")

# ===============================
# RANDOM FOREST REGRESSION
# ===============================
st.subheader("🌲 Random Forest Regression")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

st.write(f"**R² Random Forest:** {r2_rf:.3f}")
st.write(f"**RMSE Random Forest:** {rmse_rf:.3f}")

# ===============================
# PERBANDINGAN PREDIKSI
# ===============================
st.subheader("📊 Perbandingan Prediksi vs Aktual")

fig_p, ax_p = plt.subplots(1, 2, figsize=(12, 5))

ax_p[0].scatter(y_test, y_pred_lr)
ax_p[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax_p[0].set_title("Regresi Linear")
ax_p[0].set_xlabel("Aktual")
ax_p[0].set_ylabel("Prediksi")

ax_p[1].scatter(y_test, y_pred_rf)
ax_p[1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax_p[1].set_title("Random Forest")
ax_p[1].set_xlabel("Aktual")
ax_p[1].set_ylabel("Prediksi")

st.pyplot(fig_p)

st.success("✅ Aplikasi berjalan dengan baik tanpa error")




