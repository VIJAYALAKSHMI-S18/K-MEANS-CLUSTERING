import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load data and saved models
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🛍️ Mall Customer Segmentation (K-Means)")
st.write("Clustering customers based on **Annual Income** and **Spending Score**")

# -----------------------------
# Show Dataset
# -----------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# -----------------------------
# Cluster Visualization
# -----------------------------
st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in range(kmeans.n_clusters):
    ax.scatter(
        df[df["Cluster"] == cluster]["Annual Income (k$)"],
        df[df["Cluster"] == cluster]["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Predict Customer Cluster")

income = st.number_input("Annual Income (k$)", min_value=0, value=50)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):
    new_data = scaler.transform([[income, score]])
    prediction = kmeans.predict(new_data)[0]
    st.success(f"✅ Customer belongs to **Cluster {prediction}**")
