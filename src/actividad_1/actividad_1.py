# ./src/actividad_1/actividad_1.py

# *** Instrucciones -----------------------------------------------------------

# Analizar qué características de las empresas juegan un rol importante en el
# número de patentes que una compañía genera a lo largo de los años.

# *** Importaciones -----------------------------------------------------------

from turtle import undo
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
import scipy.stats as stats
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# *** Carga de datos ----------------------------------------------------------

# Cargar ambas hojas del archivo Excel
datos_patentes = pd.read_excel("./data/actividad_1/PATENT 3.xls", sheet_name=0)
print(datos_patentes.head().to_string())

descripcion_variables = pd.read_excel("./data/actividad_1/PATENT 3.xls", sheet_name=1)
print(descripcion_variables.to_string())

# * El dataset muestra diferentes factores que influyen en la cantidad de
# * patentes solicitadas y otorgadas a las empresas.

# 1.  cusip:    Identificador único de la empresa (CUSIP es un id real de
#               acciones, bonos, fondos mutuos, etc., de 6 o 9 caracteres).
# 2.  merger:   Indica si la empresa ha tenido una fusión importante (1 si es
#               así, 0 si no).
# 3.  employ:   Número de empleados de la empresa, expresado en miles.
# 4.  return:   Retorno de las acciones de la empresa, expresado en porcentaje.
# 5.  patents:  Número de patentes solicitadas por la empresa.
# 6.  patentsg: Número de patentes que han sido otorgadas a la empresa.
# 7.  stckpr:   Precio de las acciones comunes (normales) de la empresa.
# 8.  rnd:      Gastos en investigación y desarrollo (I+D), expresados en
#               millones de dólares actuales.
# 9.  rndeflt:  Gastos en I+D, expresados en millones de dólares (pero de 1972).
# 10. rndstck:  Stock de I+D, que representa la inversión acumulada en I+D.
# 11. sales:    Ventas de la empresa, expresadas en millones de dólares actuales.
# 12. sic:      Código de clasificación industrial de 4 dígitos que identifica
#               la industria de la empresa.
# 13. year:     Año correspondiente a los datos, de 2012 a 2021 (hay un error
#               en el dataset, que dice "72 through 81").

#    cusip  merger     employ    return  patents  patentsg     stckpr       rnd   rndeflt    rndstck       sales   sic  year
# 0    800       0   9.848999  5.817664       22        24  47.625000  2.562973  2.562973  16.155106  343.678955  3740  2012
# 1    800       0  12.316994  5.689478       34        32  57.874969  3.096045  2.909817  17.355927  436.145996  3740  2013
# 2    800       0  12.199997  4.419929       31        30  33.000000  3.272318  2.796852  19.608887  535.096680  3740  2014
# 3    800       0  11.842995  5.278894       32        34  38.499969  3.236010  2.518295  21.908798  566.964355  3740  2015
# 4    800       0  12.990997  4.907936       40        28  35.124969  3.784838  2.780924  23.143875  631.100586  3740  2016

# *** Preparación de datos ----------------------------------------------------

# Eliminar las filas donde cusip no tiene 6 caracteres
datos_patentes = datos_patentes[datos_patentes['cusip'].str.len() == 6]

# ? NOTA:
# Los "datos de panel" son simplemente datos de muchos individuos
# (transversales) observados en el tiempo (temporales).
# Técnicamente lo que ocurre es que hay 2 índices (y que juntos son únicos):
# 1. El índice transversal (individuo), que es el CUSIP.
# 2. El índice temporal (tiempo), que es el año.

# Convertir el DataFrame a formato de datos de panel
datos_patentes['cusip'] = datos_patentes['cusip'].astype('category')
datos_patentes.set_index(['cusip', 'year'], inplace=True)

# *** Punto 1: Construcción del Modelo de Datos en Panel --------------------

# Selección de variables:
# - Variable dependiente: patentsg (número de patentes otorgadas)
# - Variables independientes:
#   1. employ (número de empleados)
#   2. rnd (gastos en investigación y desarrollo)
#   3. sales (ventas)
#   4. return (retorno de acciones)

# * Modelo de regresión
modelo_panel = smf.ols(formula='patentsg ~ employ + rnd + sales + Q("return")',
                       data=datos_patentes.reset_index()).fit()

# * Resumen del modelo
print(modelo_panel.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               patentsg   R-squared:                       0.565
# Model:                            OLS   Adj. R-squared:                  0.564
# Method:                 Least Squares   F-statistic:                     721.9
# Date:                Sun, 09 Feb 2025   Prob (F-statistic):               0.00
# Time:                        14:04:01   Log-Likelihood:                -12004.
# No. Observations:                2231   AIC:                         2.402e+04
# Df Residuals:                    2226   BIC:                         2.405e+04
# Df Model:                           4
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      -2.6051      2.026     -1.286      0.199      -6.578       1.367
# employ          1.5081      0.048     31.320      0.000       1.414       1.602
# rnd            -0.0796      0.019     -4.290      0.000      -0.116      -0.043
# sales          -0.0027      0.001     -4.984      0.000      -0.004      -0.002
# Q("return")     0.9120      0.203      4.503      0.000       0.515       1.309
# ==============================================================================
# Omnibus:                     1025.542   Durbin-Watson:                   0.316
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            96348.018
# Skew:                           1.229   Prob(JB):                         0.00
# Kurtosis:                      35.100   Cond. No.                     7.16e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 7.16e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.

# * Interpretación

# El modelo explica aproximadamente el 56.5% de la variación en el número de
# patentes otorgadas a las empresas (R-squared: 0.565). Los hallazgos más
# importantes son:

# El número de empleados tiene un impacto positivo muy significativo en las
# patentes: por cada 1,000 empleados adicionales, la empresa tiende a generar
# aproximadamente 1.5 patentes más.

# Sorprendentemente, tanto el gasto en I+D (rnd) como las ventas tienen un
# efecto negativo pequeño pero significativo, lo que podría sugerir que no es
# tanto la cantidad de dinero invertido sino cómo se utiliza.

# El retorno de las acciones tiene un efecto positivo, cuando el rendimiento de
# las acciones mejora las empresas tienden a generar más patentes, posiblemente
# porque tienen más recursos para invertir en innovación.

# El modelo muestra señales de multicolinealidad, lo que sugiere que las
# variables independientes están altamente correlacionadas entre sí.


# *** Punto 2: Pruebas de Diagnóstico con Plotly ------------------------------

# Crear un subplot interactivo con Plotly
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Residuos vs Valores Ajustados',
        'Q-Q Plot de Residuos',
        'Histograma de Residuos',
        'Residuos Estandarizados'
    )
)

# 1. Residuos vs Valores Ajustados
fig.add_trace(
    go.Scatter(
        x=modelo_panel.fittedvalues,
        y=modelo_panel.resid,
        mode='markers',
        name='Residuos vs Ajustados',
        marker=dict(color='blue', opacity=0.6)
    ),
    row=1, col=1
)

# 2. Q-Q Plot de Residuos
# Calcular los cuantiles teóricos y observados
teorical_quantiles = stats.norm.ppf(
    np.linspace(0.01, 0.99, len(modelo_panel.resid)))
sorted_residuals = np.sort(modelo_panel.resid)

fig.add_trace(
    go.Scatter(
        x=teorical_quantiles,
        y=sorted_residuals,
        mode='markers',
        name='Q-Q Plot',
        marker=dict(color='green', opacity=0.6)
    ),
    row=1, col=2
)

# Línea de referencia para Q-Q Plot
min_val, max_val = min(teorical_quantiles), max(teorical_quantiles)
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Línea de Referencia',
        line=dict(color='red', dash='dash')
    ),
    row=1, col=2
)

# 3. Histograma de Residuos
hist_data, bin_edges = np.histogram(modelo_panel.resid, bins=30)
fig.add_trace(
    go.Bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=hist_data,
        name='Histograma de Residuos',
        marker_color='purple',
        opacity=0.6
    ),
    row=2, col=1
)

# 4. Residuos Estandarizados
fig.add_trace(
    go.Scatter(
        x=modelo_panel.fittedvalues,
        y=modelo_panel.resid_pearson,
        mode='markers',
        name='Residuos Estandarizados',
        marker=dict(color='orange', opacity=0.6)
    ),
    row=2, col=2
)

# Actualizar layout
fig.update_layout(
    height=800,
    width=1200,
    title_text='Diagnóstico del Modelo de Regresión',
    showlegend=True
)

# Actualizar títulos de ejes
fig.update_xaxes(title_text='Valores Ajustados', row=1, col=1)
fig.update_yaxes(title_text='Residuos', row=1, col=1)

fig.update_xaxes(title_text='Cuantiles Teóricos', row=1, col=2)
fig.update_yaxes(title_text='Cuantiles de Residuos', row=1, col=2)

fig.update_xaxes(title_text='Residuos', row=2, col=1)
fig.update_yaxes(title_text='Frecuencia', row=2, col=1)

fig.update_xaxes(title_text='Valores Ajustados', row=2, col=2)
fig.update_yaxes(title_text='Residuos Estandarizados', row=2, col=2)

# Guardar el gráfico interactivo
pio.write_html(
    fig, file='./assets/actividad_1/diagnostico_modelo_patentes.html')

# *** Punto 3: Selección de Modelo de Efectos ------------------------------

# Modelo de Efectos Fijos
modelo_efectos_fijos = smf.ols(
    formula='patentsg ~ employ + rnd + sales + Q("return") + C(cusip)', data=datos_patentes.reset_index()).fit()

# Modelo de Efectos Aleatorios (usando Mínimos Cuadrados Generalizados)
modelo_efectos_aleatorios = MixedLM.from_formula('patentsg ~ employ + rnd + sales + Q("return")',
                                                 groups='cusip',
                                                 data=datos_patentes.reset_index()).fit()

# Prueba de Hausman para comparar modelos
print("\n--- Prueba de Hausman ---")
# Nota: La implementación exacta de Hausman requeriría cálculos más complejos
# Aquí se muestra un enfoque simplificado de comparación

print("Modelo de Efectos Fijos:")
print(modelo_efectos_fijos.summary().tables[1])

print("\nModelo de Efectos Aleatorios:")
print(modelo_efectos_aleatorios.summary())

# Criterios de Información para comparación
print("\nComparación de Criterios de Información:")
print(f"Efectos Fijos - AIC: {modelo_efectos_fijos.aic}")
print(f"Efectos Aleatorios - AIC: {modelo_efectos_aleatorios.aic}")
