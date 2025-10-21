# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Загрузка обученной модели и информации о признаках
model = joblib.load('diamond_price_model.pkl')


# Определение всех признаков (соответствует обучению модели)
all_columns = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Fair',
       'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_D',
       'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
       'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
       'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2']

# Заголовок приложения
st.title('Прогноз стоимости бриллиантов')

# Ввод данных пользователя
carat = st.number_input('Карат', min_value=0.0, max_value=5.0, value=1.0, step=0.01)
depth = st.number_input('Глубина (%)', min_value=40.0, max_value=80.0, value=60.0, step=0.1)
table = st.number_input('Площадка (%)', min_value=40.0, max_value=80.0, value=55.0, step=0.1)
x = st.number_input('Размер x (мм)', min_value=0.0, max_value=15.0, value=4.0, step=0.1)
y = st.number_input('Размер y (мм)', min_value=0.0, max_value=15.0, value=4.0, step=0.1)
z = st.number_input('Размер z (мм)', min_value=0.0, max_value=15.0, value=2.5, step=0.1)
cut = st.selectbox('Оценка огранки', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Цвет', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Чистота', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Преобразование введенных данных в формат, понятный модели
input_data = {
    'carat': carat,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z,
    'cut_' + cut: 1,
    'color_' + color: 1,
    'clarity_' + clarity: 1
}

# Заполняем отсутствующие признаки нулями
input_df = pd.DataFrame([input_data], columns=all_columns).fillna(0)

# Предсказание стоимости
if st.button('Предсказать стоимость'):
    prediction = model.predict(input_df)
    st.success(f'Ориентировочная стоимость бриллианта: ${prediction[0]:.2f}')
