import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)

st.sidebar.title('Controles Sidebar')

age = st.sidebar.slider('Edad en la empresa',0,50,25)

st.markdown(
    """
    # Reporte de ventas de Zapatos

    Este reporte muestra las ventas de zapatos durante un año.
    En el reporet de identifica un crecimiento de ventas de zapatos en el mes de diciembre.

    Estos datos, indican ventas de zapatoss por años de venta
    """
)

x = np.linspace(0, 50, 51)

y = x + 10 * np.random.random(len(x))

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

st.plotly_chart(fig)

st.markdown(
    """
    # Modelado de datos
    En los datos se observa un comportamiento lineal
    por lo que se realiza un ajusta de una linea recta con una regresion lineal

    $$y = m*x + b$$
    """
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mse')

history = model.fit(x,y, epochs=2000, verbose=0)

st.write('Perdida final: ', history.history['loss'][-1])

x_loss = np.linspace(1, len(history.history['loss']), 
                     len(history.history['loss']))
y_loss = history.history['loss']

fig = plt.figure()
plt.plot(x_loss,y_loss)
st.pyplot(fig)

y_pred = model.predict(x)
y_pred = y_pred.reshape(-1)

fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x=x, y=y, mode='markers'))
fig_2.add_trace(go.Scatter(x=x, y=y_pred, mode='lines'))
st.plotly_chart(fig_2)

st.write("La edad en la empresa es:", age)

expected_val = model.predict([age])

st.write('Se espera vender: ', expected_val)

st.latex(
    r"""
    f(x) = \dfrac{1}{1+e^{-x}}
    """
)

st.latex(
    r"""
    e = mc^2
    """
)
