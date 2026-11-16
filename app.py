import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Amazon Sales Revenue Predictor", page_icon="🛒", layout="wide")

DATA_FILE = "amazon_sales_dataset.csv"


@st.cache_resource(show_spinner="Training Random Forest model...")
def load_model():
    df = pd.read_csv(DATA_FILE)
    df_model = df.drop(columns=["order_id", "order_date", "product_id"])
    df_encoded = pd.get_dummies(
        df_model,
        columns=["product_category", "customer_region", "payment_method"],
        drop_first=True,
    )
    X = df_encoded.drop(columns=["total_revenue"])
    y = df_encoded["total_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return df, model, X.columns, metrics, importances, (y_test, y_pred)


df, model, feature_cols, metrics, importances, (y_test, y_pred) = load_model()


def predict_revenue(price, discount, qty, rating, reviews, category, region, payment):
    discounted_price = round(price * (1 - discount / 100), 2)
    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    row["price"] = price
    row["discount_percent"] = discount
    row["quantity_sold"] = qty
    row["rating"] = rating
    row["review_count"] = reviews
    row["discounted_price"] = discounted_price
    for col, val in {
        "product_category": category,
        "customer_region": region,
        "payment_method": payment,
    }.items():
        name = f"{col}_{val}"
        if name in row.columns:
            row[name] = 1
    return float(model.predict(row)[0]), discounted_price


st.title("🛒 Amazon Sales Total Revenue Predictor")
st.write(
    "Enter product and sales details to predict **total revenue** using a "
    "Random Forest Regressor trained on 50,000 Amazon orders."
)

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Data Explorer", "🤖 Model Performance"])

with tab_predict:
    c1, c2 = st.columns(2)
    with c1:
        price = st.number_input("Price ($)", min_value=0.0, value=150.0, step=1.0)
        discount = st.slider("Discount Percent (%)", 0, 50, 10)
        qty = st.number_input("Quantity Sold", min_value=1, value=3, step=1)
        rating = st.slider("Rating", 1.0, 5.0, 4.0, step=0.1)
        reviews = st.number_input("Review Count", min_value=0, value=250, step=10)
    with c2:
        category = st.selectbox("Product Category", sorted(df["product_category"].unique()))
        region = st.selectbox("Customer Region", sorted(df["customer_region"].unique()))
        payment = st.selectbox("Payment Method", sorted(df["payment_method"].unique()))
        st.caption("Discounted price is calculated automatically: price × (1 − discount%).")

    if st.button("Predict Revenue", type="primary"):
        pred, disc_price = predict_revenue(
            price, discount, qty, rating, reviews, category, region, payment
        )
        m1, m2 = st.columns(2)
        m1.metric("Discounted Price", f"${disc_price:,.2f}")
        m2.metric("Predicted Total Revenue", f"${pred:,.2f}")

with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Revenue by Product Category")
    st.bar_chart(df.groupby("product_category")["total_revenue"].sum())
    st.subheader("Revenue by Customer Region")
    st.bar_chart(df.groupby("customer_region")["total_revenue"].sum())
    st.subheader("Revenue by Payment Method")
    st.bar_chart(df.groupby("payment_method")["total_revenue"].sum())

with tab_model:
    m1, m2 = st.columns(2)
    m1.metric("R² Score (test set)", f"{metrics['r2']:.4f}")
    m2.metric("RMSE (test set)", f"{metrics['rmse']:.4f}")

    st.subheader("Top 10 Important Features")
    st.bar_chart(importances.nlargest(10))

    st.subheader("Actual vs Predicted (sample of 1,000 test rows)")
    sample = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).sample(1000, random_state=42)
    st.scatter_chart(sample, x="Actual", y="Predicted")
