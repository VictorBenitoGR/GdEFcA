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

# *** Preparación de datos ----------------------------------------------------

# * Carga de datos
datos_patentes = pd.read_excel("./data/actividad_1/PATENT 3.xls", sheet_name=0)
print(datos_patentes.head().to_string())

descripcion_variables = pd.read_excel(
    "./data/actividad_1/PATENT 3.xls", sheet_name=1)
print(descripcion_variables.to_string())

# * Descripción de las variables
# El dataset muestra diferentes factores que influyen en la cantidad de
# patentes solicitadas y otorgadas a las empresas.

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

# * Limpieza de datos
# Eliminar las filas donde cusip no tiene 6 caracteres
datos_patentes = datos_patentes[datos_patentes['cusip'].astype(
    str).str.len() == 6]

# ? NOTA:
# Los "datos de panel" son simplemente datos de muchos individuos
# (transversales) observados en el tiempo (temporales).
# Técnicamente lo que ocurre es que hay 2 índices (y que juntos son únicos):
# 1. El índice transversal (individuo), que es el CUSIP.
# 2. El índice temporal (tiempo), que es el año.

# Convertir el DataFrame a formato de datos de panel
datos_patentes['cusip'] = datos_patentes['cusip'].astype('category')
datos_patentes.set_index(['cusip', 'year'], inplace=True)

# *** Punto 1 -----------------------------------------------------------------

# * Instrucciones
# Construye un modelo de datos en panel. Recuerda seleccionar adecuadamente tus
# variables dependiente e independientes. Recuerda que el objetivo del ejercicio
# es predecir el número de patentes que una empresa podría generar.

# * Selección de variables
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
# Dep. Variable:               patentsg   R-squared:                       0.578
# Model:                            OLS   Adj. R-squared:                  0.577
# Method:                 Least Squares   F-statistic:                     704.1
# Date:                Sun, 09 Feb 2025   Prob (F-statistic):               0.00
# Time:                        23:07:26   Log-Likelihood:                -11137.
# No. Observations:                2062   AIC:                         2.228e+04
# Df Residuals:                    2057   BIC:                         2.231e+04
# Df Model:                           4
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      -3.0918      2.134     -1.449      0.148      -7.277       1.093
# employ          1.5993      0.051     31.646      0.000       1.500       1.698
# rnd            -0.1221      0.020     -6.211      0.000      -0.161      -0.084
# sales          -0.0023      0.001     -4.091      0.000      -0.003      -0.001
# Q("return")     1.0134      0.214      4.730      0.000       0.593       1.434
# ==============================================================================
# Omnibus:                      770.590   Durbin-Watson:                   0.323
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            78958.902
# Skew:                           0.774   Prob(JB):                         0.00
# Kurtosis:                      33.276   Cond. No.                     7.31e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 7.31e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.


# * Interpretación

# - El modelo explica aproximadamente el 57.8% de la variación en el número de
#   patentes otorgadas a las empresas (R-squared: 0.578).

# - El número de empleados tiene un impacto positivo muy significativo en las
#   patentes. Por cada 1,000 empleados adicionales, la empresa tiende a generar
#   aproximadamente 1.6 patentes más (coeficiente: 1.5993, p < 0.001).

# - Sorprendentemente el gasto en I+D tiene un efecto negativo significativo en
#   el número de patentes otorgadas. Esto sugiere que no es solo la cantidad de
#   dinero invertido en I+D lo que importa, sino cómo se utiliza (coeficiente:
#   -0.1221, p < 0.001).

# - Las ventas también tienen un efecto negativo pequeño pero significativo en
#   el número de patentes otorgadas (coeficiente: -0.0023, p < 0.001). Esto
#   podría indicar que un aumento en las ventas no necesariamente se traduce en
#   un aumento en la innovación.

# - El retorno de las acciones tiene un efecto positivo significativo. Cuando el
#   rendimiento de las acciones mejora, las empresas tienden a generar más
#   patentes, posiblemente porque tienen más recursos para invertir en
#   innovación (coeficiente: 1.0134, p < 0.001).

# - El modelo muestra señales de multicolinealidad, sugiriendo que las variables
#   independientes están altamente correlacionadas entre sí. Esto puede afectar
#   la estabilidad de los coeficientes estimados y la interpretación de los
#   resultados.

# *** Punto 2 -----------------------------------------------------------------

# * Instrucciones
# Realiza las pruebas necesarias para detectar posibles errores en tu análisis.
# Verifica la presencia o ausencia de heterocedasticidad y autocorrelación
# serial.

# * Inicialización
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '<b>Residuos vs Valores ajustados</b>',
        '<b>Q-Q Plot de residuos</b>',
        '<b>Histograma de residuos</b>',
        '<b>Residuos estandarizados</b>'
    )
)

# * Residuos vs Valores Ajustados
fig.add_trace(
    go.Scatter(
        x=modelo_panel.fittedvalues,
        y=modelo_panel.resid,
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        showlegend=False
    ),
    row=1, col=1
)

# * Q-Q Plot de residuos
# Calcular los cuantiles teóricos y observados
residuos = modelo_panel.resid
sorted_residuals = np.sort(residuos)
teorical_quantiles = stats.norm.ppf(
    np.linspace(0.01, 0.99, len(sorted_residuals)))

fig.add_trace(
    go.Scatter(
        x=teorical_quantiles,
        y=sorted_residuals,
        mode='markers',
        marker=dict(color='green', opacity=0.6),
        showlegend=False
    ),
    row=1, col=2
)

# * Línea de referencia para Q-Q Plot
min_val, max_val = min(teorical_quantiles), max(teorical_quantiles)

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ),
    row=1, col=2
)

# * Histograma de residuos
hist_data, bin_edges = np.histogram(modelo_panel.resid, bins=30)

fig.add_trace(
    go.Bar(
        x=(bin_edges[:-1] + bin_edges[1:]) / 2,
        y=hist_data,
        marker_color='purple',
        opacity=0.6,
        showlegend=False
    ),
    row=2, col=1
)

# * Residuos estandarizados
fig.add_trace(
    go.Scatter(
        x=modelo_panel.fittedvalues,
        y=modelo_panel.resid_pearson,
        mode='markers',
        marker=dict(color='orange', opacity=0.6),
        showlegend=False
    ),
    row=2, col=2
)

# * Actualización de layout
fig.update_layout(
    height=800,  # Altura fija en píxeles en lugar de vh
    title_text='<b>Diagnóstico del modelo de regresión</b>',
    title_font=dict(size=24),
    showlegend=False,
    autosize=True,
    margin=dict(l=40, r=40, t=80, b=40),
    grid=dict(
        rows=2,
        columns=2,
        pattern='independent',
        roworder='top to bottom',
        ygap=0.1,
        xgap=0.1
    ),
)

# * Títulos de ejes
fig.update_xaxes(title_text='Valores ajustados', row=1, col=1)
fig.update_yaxes(title_text='Residuos', row=1, col=1)

fig.update_xaxes(title_text='Cuantiles teóricos', row=1, col=2)
fig.update_yaxes(title_text='Cuantiles de residuos', row=1, col=2)

fig.update_xaxes(title_text='Residuos', row=2, col=1)
fig.update_yaxes(title_text='Frecuencia', row=2, col=1)

fig.update_xaxes(title_text='Valores ajustados', row=2, col=2)
fig.update_yaxes(title_text='Residuos estandarizados', row=2, col=2)

# * Exportar
pio.write_html(
    fig,
    file='./docs/actividad_1/diagnostico_modelo_patentes.html',
    config={
        'responsive': True,
        'displayModeBar': True,
        'scrollZoom': True
    },
    full_html=True,
    include_plotlyjs=True,
    validate=True,
)


# *** Punto 3 -----------------------------------------------------------------

# * Instrucciones
# Determina cuál es el modelo apropiado de datos panel para este caso (efectos
# fijos o efectos aleatorios).

# Preparar los datos
datos_panel = datos_patentes.reset_index()

# Modelo de Efectos Fijos
modelo_efectos_fijos = smf.ols(
    formula='patentsg ~ employ + rnd + sales + Q("return") + C(cusip)',
    data=datos_panel).fit()

# Modelo de Efectos Aleatorios
try:
    modelo_efectos_aleatorios = MixedLM.from_formula(
        'patentsg ~ employ + rnd + sales + Q("return")',
        groups=datos_panel['cusip'],
        data=datos_panel,
        re_formula='1'  # Término aleatorio para el intercepto
    ).fit()
except:
    print("No se pudo ajustar el modelo de efectos aleatorios")
    modelo_efectos_aleatorios = None

# Mostrar resultados
print("\n--- Resultados del Modelo de Efectos Fijos ---")
print(modelo_efectos_fijos.summary().tables[1])

if modelo_efectos_aleatorios is not None:
    print("\n--- Resultados del Modelo de Efectos Aleatorios ---")
    print(modelo_efectos_aleatorios.summary())

# Criterios de Información para comparación
print("\nComparación de Criterios de Información:")
print(f"Efectos Fijos - AIC: {modelo_efectos_fijos.aic}")
if modelo_efectos_aleatorios is not None:
    print(f"Efectos Aleatorios - AIC: {modelo_efectos_aleatorios.aic}")

# *** Punto 4 -----------------------------------------------------------------

# * Instrucciones
# Interpreta los resultados y comenta que tan buenos serían los pronósticos
# generados con el modelo propuesto.
