import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Aplikasi ini menampilkan clustering provinsi dan regresi linear COVID-19.")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Covid-19_Indonesia_Dataset.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y', errors='coerce')
    return df

df = load_data()

# ===============================
# FILTER DATA
# ===============================
tanggal = st.date_input("📅 Pilih Tanggal", value=pd.to_datetime("2022-09-15"))
df_cluster = df[df['Tanggal'] == pd.to_datetime(tanggal)].copy()
df_cluster = df_cluster[df_cluster['Provinsi'].notnull()]

# ===============================
# FITUR
# ===============================
df_cluster = df_cluster[
    ['Provinsi','Total_Kasus','Total_Kematian','Total_Sembuh',
     'Populasi','Kepadatan_Penduduk',
     'Total_Kasus_Per_Juta','Total_Kematian_Per_Juta']
]

df_cluster['Rasio_Kematian'] = df_cluster['Total_Kematian'] / df_cluster['Total_Kasus']
df_cluster['Rasio_Kesembuhan'] = df_cluster['Total_Sembuh'] / df_cluster['Total_Kasus']
df_cluster.fillna(0, inplace=True)

fitur = [
    'Total_Kasus','Total_Kematian','Total_Sembuh','Populasi',
    'Kepadatan_Penduduk','Total_Kasus_Per_Juta',
    'Total_Kematian_Per_Juta','Rasio_Kematian','Rasio_Kesembuhan'
]

# ===============================
# SCALING
# ===============================
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df_cluster[fitur])

# ===============================
# CLUSTERING
# ===============================
st.subheader("🔹 Clustering Provinsi")

k = st.slider("Jumlah Cluster (k)", 2, 7, 5)
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_df)

# ===============================
# VISUALISASI CLUSTER
# ===============================
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(
    data=df_cluster,
    x='Kepadatan_Penduduk',
    y='Total_Kematian_Per_Juta',
    hue='Cluster',
    palette='viridis',
    s=100,
    ax=ax
)
ax.set_title("Sebaran Cluster Provinsi")
ax.grid(True)
st.pyplot(fig)

# ===============================
# RINGKASAN CLUSTER
# ===============================
st.subheader("📌 Karakteristik Rata-rata Cluster")
summary = df_cluster.groupby('Cluster')[fitur].mean()
st.dataframe(summary)

# ===============================
# REGRESI LINEAR
# ===============================
st.subheader("📈 Regresi Linear")

X = df_cluster[['Populasi','Kepadatan_Penduduk','Total_Kasus_Per_Juta']]
y = df_cluster['Total_Kematian_Per_Juta']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**R² Score:** {r2:.3f}")
st.write(f"**RMSE:** {rmse:.3f}")

# ===============================
# PREDIKSI VS AKTUAL
# ===============================
fig2, ax2 = plt.subplots(figsize=(6,5))
ax2.scatter(y_test, y_pred)
ax2.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
ax2.set_xlabel("Aktual")
ax2.set_ylabel("Prediksi")
ax2.set_title("Prediksi vs Aktual")
ax2.grid(True)
st.pyplot(fig2)

st.success("✅ Aplikasi berhasil dijalankan tanpa error")
