import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Heart Disease Prediction (Cleveland) на Streamlit")
st.write(
    """
Используем датасет **Cleveland Heart Disease (processed.cleveland.data)**  
и обучаем простую модель классификации (Random Forest), 
чтобы предсказать наличие сердечного заболевания.
"""
)

# ---------- 1. Загрузка и базовая обработка данных ----------

@st.cache_data
def load_raw_data():
    """
    Читаем файл processed.cleveland.data без заголовков.
    В файле пропуски помечены знаком '?'.
    """
    column_names = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"
    ]

    df = pd.read_csv(
        "processed.cleveland.data",
        header=None,
        names=column_names,
        na_values="?"
    )

    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    """
    - Убираем строки с пропусками
    - Делаем бинарную цель: target = (num > 0)
    - One-hot кодирование категориальных признаков
    """
    df_clean = df.dropna().copy()

    # Бинарная цель: 1 - есть болезнь, 0 - нет
    df_clean["target"] = (df_clean["num"] > 0).astype(int)
    df_clean = df_clean.drop(columns=["num"])

    # Числовые и категориальные признаки
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    X = df_clean[numeric_features + categorical_features]
    y = df_clean["target"]

    # One-hot для категориальных
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    return X, X_encoded, y, numeric_features, categorical_features, X_encoded.columns


@st.cache_resource
def train_model(X_encoded, y):
    """
    Делим на train/test, обучаем RandomForest, считаем accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


df_raw = load_raw_data()
st.subheader("Первые строки данных")
st.dataframe(df_raw.head())

X_raw, X_encoded, y, numeric_features, categorical_features, model_feature_cols = preprocess_data(df_raw)
model, acc = train_model(X_encoded, y)

st.subheader("Качество модели")
st.write(f"Accuracy на тестовой выборке: **{acc:.3f}**")

st.markdown("---")
st.header("🔮 Предсказание для нового пациента")

# ---------- 2. Форма ввода признаков ----------

st.sidebar.header("Параметры пациента")

# Диапазоны для слайдеров берём из реальных данных
def num_range(col):
    return float(X_raw[col].min()), float(X_raw[col].max()), float(X_raw[col].mean())

age_min, age_max, age_mean = num_range("age")
trestbps_min, trestbps_max, trestbps_mean = num_range("trestbps")
chol_min, chol_max, chol_mean = num_range("chol")
thalach_min, thalach_max, thalach_mean = num_range("thalach")
oldpeak_min, oldpeak_max, oldpeak_mean = num_range("oldpeak")

# Числовые признаки
age = st.sidebar.slider("Возраст (age)", int(age_min), int(age_max), int(age_mean))
trestbps = st.sidebar.slider("Рез. давление (trestbps)", int(trestbps_min), int(trestbps_max), int(trestbps_mean))
chol = st.sidebar.slider("Холестерин (chol)", int(chol_min), int(chol_max), int(chol_mean))
thalach = st.sidebar.slider("Макс. ЧСС (thalach)", int(thalach_min), int(thalach_max), int(thalach_mean))
oldpeak = st.sidebar.slider("oldpeak (депрессия ST)", float(oldpeak_min), float(oldpeak_max), float(oldpeak_mean))

# Категориальные (берём уникальные значения из данных)
def cat_options(col):
    return sorted(X_raw[col].astype(int).unique().tolist())

sex = st.sidebar.selectbox("Пол (sex: 0=жен, 1=муж)", options=cat_options("sex"))
cp = st.sidebar.selectbox("Тип боли в груди (cp 1-4)", options=cat_options("cp"))
fbs = st.sidebar.selectbox("fbs > 120 (0/1)", options=cat_options("fbs"))
restecg = st.sidebar.selectbox("restecg (0-2)", options=cat_options("restecg"))
exang = st.sidebar.selectbox("Нагрузочная стенокардия exang (0/1)", options=cat_options("exang"))
slope = st.sidebar.selectbox("slope (1=up,2=flat,3=down)", options=cat_options("slope"))
ca = st.sidebar.selectbox("Кол-во сосудов ca (0-3)", options=cat_options("ca"))
thal = st.sidebar.selectbox("thal (3=norm,6=fixed,7=reversible)", options=cat_options("thal"))

# Собираем "сырые" признаки в DataFrame
input_raw = pd.DataFrame([{
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak,
    "sex": sex,
    "cp": cp,
    "fbs": fbs,
    "restecg": restecg,
    "exang": exang,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])


def encode_input(input_raw_df, categorical_features, model_feature_cols):
    """
    Применяем те же преобразования, что и к обучающим данным:
    - get_dummies по тем же категориальным
    - reindex к колонкам обучающей выборки
    """
    input_encoded = pd.get_dummies(
        input_raw_df,
        columns=categorical_features,
        drop_first=True
    )
    input_encoded = input_encoded.reindex(columns=model_feature_cols, fill_value=0)
    return input_encoded


if st.button("Сделать предсказание"):
    input_for_model = encode_input(input_raw, categorical_features, model_feature_cols)
    proba = model.predict_proba(input_for_model)[0]
    pred = model.predict(input_for_model)[0]

    st.subheader("Результат предсказания")
    label = "Есть признаки болезни сердца" if pred == 1 else "Признаков болезни сердца нет"
    st.write(f"**Класс:** {label}")

    st.write("Вероятности классов (0 = нет болезни, 1 = есть болезнь):")
    proba_df = pd.DataFrame([proba], columns=["0 (нет болезни)", "1 (болезнь)"])
    st.dataframe(proba_df.style.format("{:.3f}"))
else:
    st.info("Заполните параметры в сайдбаре и нажмите **Сделать предсказание**.")
