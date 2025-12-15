import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Clustering provinsi dan perbandingan Regresi Linear vs Decision Tree.")

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
# PILIH TANGGAL
# ===============================
tanggal = st.date_input(
    "📅 Pilih Tanggal",
    value=pd.to_datetime("2022-09-15")
)

df_cluster = df[df['Tanggal'] == pd.to_datetime(tanggal)].copy()
df_cluster = df_cluster[df_cluster['Provinsi'].notnull()]

# ===============================
# PILIH KOLOM
# ===============================
kolom = [
    'Provinsi','Total_Kasus','Total_Kematian','Total_Sembuh',
    'Populasi','Kepadatan_Penduduk',
    'Total_Kasus_Per_Juta','Total_Kematian_Per_Juta'
]
df_cluster = df_cluster[kolom]

# ===============================
# FITUR TURUNAN
# ===============================
df_cluster['Rasio_Kematian'] = df_cluster['Total_Kematian'] / df_cluster['Total_Kasus']
df_cluster['Rasio_Kesembuhan'] = df_cluster['Total_Sembuh'] / df_cluster['Total_Kasus']

# ===============================
# FITUR NUMERIK
# ===============================
fitur = [
    'Total_Kasus','Total_Kematian','Total_Sembuh','Populasi',
    'Kepadatan_Penduduk','Total_Kasus_Per_Juta',
    'Total_Kematian_Per_Juta','Rasio_Kematian','Rasio_Kesembuhan'
]

df_cluster[fitur] = df_cluster[fitur].apply(pd.to_numeric, errors='coerce')
df_cluster.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cluster.dropna(subset=fitur, inplace=True)

# ===============================
# VALIDASI DATA
# ===============================
if df_cluster.shape[0] < 3:
    st.warning("⚠️ Data provinsi terlalu sedikit, silakan pilih tanggal lain.")
    st.stop()

if (df_cluster[fitur].std() == 0).any():
    st.warning("⚠️ Salah satu fitur tidak memiliki variasi nilai.")
    st.stop()

# ===============================
# MATRKS DATA (UNTUK LAPORAN)
# ===============================
st.subheader("📐 Matriks Data (Input Model)")

X_matrix = df_cluster[['Populasi','Kepadatan_Penduduk','Total_Kasus_Per_Juta']]
y_vector = df_cluster['Total_Kematian_Per_Juta']

st.write("**Matriks X (Fitur)**")
st.dataframe(X_matrix.head())

st.write("**Vektor y (Target)**")
st.dataframe(y_vector.head())

# ===============================
# SCALING & CLUSTERING
# ===============================
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df_cluster[fitur])

st.subheader("🔹 Clustering Provinsi (K-Means)")
k = st.slider("Jumlah Cluster (k)", 2, 7, 5)

kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_df)

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
# REGRESI: TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y_vector, test_size=0.2, random_state=42
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

st.write(f"**R² Score (Linear):** {r2_lr:.3f}")
st.write(f"**RMSE (Linear):** {rmse_lr:.3f}")

# ===============================
# DECISION TREE REGRESSION
# ===============================
st.subheader("🌳 Decision Tree Regression")

tree = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

r2_tree = r2_score(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))

st.write(f"**R² Score (Decision Tree):** {r2_tree:.3f}")
st.write(f"**RMSE (Decision Tree):** {rmse_tree:.3f}")

# ===============================
# VISUALISASI PERBANDINGAN
# ===============================
fig2, ax2 = plt.subplots(figsize=(6,5))
ax2.scatter(y_test, y_pred_tree)
ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
ax2.set_xlabel("Aktual")
ax2.set_ylabel("Prediksi")
ax2.set_title("Decision Tree: Prediksi vs Aktual")
ax2.grid(True)
st.pyplot(fig2)

# ===============================
# VISUALISASI POHON KEPUTUSAN
# ===============================
st.subheader("🌲 Visualisasi Pohon Keputusan")

fig3, ax3 = plt.subplots(figsize=(18,6))
plot_tree(
    tree,
    feature_names=X_matrix.columns,
    filled=True,
    max_depth=3,
    fontsize=9
)
st.pyplot(fig3)

st.success("✅ Aplikasi berjalan normal (Linear + Decision Tree + Matriks)")



