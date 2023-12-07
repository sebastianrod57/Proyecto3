import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import psycopg2
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

def grafica1():
    # Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="icfes",
        user="postgres",
        password="Proyecto3",
        host="proyecto3-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port='5432'
    )

    # Consulta SQL
    consulta1 = """
    SELECT
        Bilingue AS colegio_bilingue,
        Puntaje_Ingles AS rango_puntaje,
        COUNT(*) AS cantidad_estudiantes
    FROM icfes
    GROUP BY colegio_bilingue, rango_puntaje
    ORDER BY colegio_bilingue, rango_puntaje;
    """

    # Crear DataFrame con los datos de la consulta
    df = pd.read_sql_query(consulta1, engine)

    # Lo siguiente es para obtener "No Bilingue" y "Bilingue" en vez de 0 y 1
    df['colegio_bilingue'] = df['colegio_bilingue'].map({0: 'No', 1: 'Si'})

    # Filtrar datos para dos gráficas separadas
    df_bilingue = df[df['colegio_bilingue'] == 'Si']
    df_no_bilingue = df[df['colegio_bilingue'] == 'No']

    # Gráfica para colegios bilingües
    fig_bilingue = px.bar(df_bilingue, x='rango_puntaje', y='cantidad_estudiantes',
                          labels={'cantidad_estudiantes': 'Número de Estudiantes'},
                          title='Cantidad de estudiantes por rango de puntaje (Colegios Bilingües)',
                          color_discrete_sequence=['#000080']
                          )

    # Gráfica para colegios no bilingües
    fig_no_bilingue = px.bar(df_no_bilingue, x='rango_puntaje', y='cantidad_estudiantes',
                             labels={'cantidad_estudiantes': 'Número de Estudiantes'},
                             title='Cantidad de estudiantes por rango de puntaje (Colegios No Bilingües)',
                             color_discrete_sequence=['#0066CC']
                             )

    # Dimensiones
    width = 800
    height = 500

    # Actualizar nombres de los ejes y márgenes
    fig_bilingue.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                               xaxis_title='Rango de Puntaje', yaxis_title='Número de Estudiantes')

    fig_no_bilingue.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                                  xaxis_title='Rango de Puntaje', yaxis_title='Número de Estudiantes')

    # Devolver el layout con las dos gráficas
    return html.Div([
        dcc.Graph(figure=fig_bilingue),
        dcc.Graph(figure=fig_no_bilingue)
    ])

def grafica2():
    # Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="icfes",
        user="postgres",
        password="Proyecto3",
        host="proyecto3-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port='5432'
    )

    # Consulta SQL
    consulta2 = """
        SELECT
        Ubicacion,
        SUM(Puntaje_Global) AS Estudiantes_Arriba,
        COUNT(*) - SUM(Puntaje_Global) AS Estudiantes_Abajo
    FROM icfes
    GROUP BY ubicacion;
    """

    # Crear DataFrame con los datos de la consulta
    df = pd.read_sql_query(consulta2, engine)

    df['ubicacion'] = df['ubicacion'].map({0: 'Rural', 1: 'Urbano'})

    # Gráfica de barras para estudiantes por encima de 350
    fig_arriba = px.bar(df, x='ubicacion', y='estudiantes_arriba',
                        labels={'value': 'Número de Estudiantes'},
                        title='Estudiantes por encima de 350 por Ubicación',
                        color_discrete_sequence=['#4B9CD3']
                        )

    # Gráfica de barras para estudiantes por debajo de 350
    fig_abajo = px.bar(df, x='ubicacion', y='estudiantes_abajo',
                       labels={'value': 'Número de Estudiantes'},
                       title='Estudiantes por debajo de 350 por Ubicación',
                       color_discrete_sequence=['#2F4E7E']
                       )

    # Dimensiones
    width = 800
    height = 500

    # Actualizar nombres de los ejes y márgenes para ambas gráficas
    fig_arriba.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                             xaxis_title='Ubicación', yaxis_title='Número de Estudiantes'
                             )

    fig_abajo.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                            xaxis_title='Ubicación', yaxis_title='Número de Estudiantes'
                            )

    # Devolver el layout con las dos gráficas
    return html.Div([
        dcc.Graph(figure=fig_arriba),
        dcc.Graph(figure=fig_abajo)
    ])

def grafica3():
    # Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="icfes",
        user="postgres",
        password="Proyecto3",
        host="proyecto3-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port='5432'
    )

    # Consulta SQL
    consulta3 = """
        SELECT
        Internet,
        SUM(Puntaje_Global) AS Estudiantes_Arriba,
        COUNT(*) - SUM(Puntaje_Global) AS Estudiantes_Abajo
    FROM icfes
    GROUP BY Internet;

    """

    # Crear DataFrame con los datos de la consulta
    df = pd.read_sql_query(consulta3, engine)

    df['internet'] = df['internet'].map({0: 'No', 1: 'Si'})

    # Gráfica de barras para estudiantes por encima de 350
    fig_arriba = px.bar(df, x='internet', y='estudiantes_arriba',
                        labels={'value': 'Número de Estudiantes'},
                        title='Estudiantes por encima de 350 con y sin Internet',
                        color_discrete_sequence=['#4B9CD3']
                        )

    # Gráfica de barras para estudiantes por debajo de 350
    fig_abajo = px.bar(df, x='internet', y='estudiantes_abajo',
                       labels={'value': 'Número de Estudiantes'},
                       title='Estudiantes por debajo de 350 con y sin Internet',
                       color_discrete_sequence=['#2F4E7E']
                       )

    # Dimensiones
    width = 800
    height = 500

    # Actualizar nombres de los ejes y márgenes para ambas gráficas
    fig_arriba.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                             xaxis_title='Internet', yaxis_title='Número de Estudiantes'
                             )

    fig_abajo.update_layout(width=width, height=height, margin=dict(l=50, r=50, b=50, t=50),
                            xaxis_title='Internet', yaxis_title='Número de Estudiantes'
                            )

    # Devolver el layout con las dos gráficas
    return html.Div([
        dcc.Graph(figure=fig_arriba),
        dcc.Graph(figure=fig_abajo)
    ])

app.layout = html.Div([

    html.H1('Herramienta de apoyo para conocer el estado de la educación en el departamento de Tolima', 
            style={'textAlign': 'center', 'color': 'black', 'backgroundColor': '#0000FF', 'padding': '20px'}),

    #Crear 4 pestañas
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Cálculo del estado de la educación', value='tab-1'),
            dcc.Tab(label='Gráficas de número de estudiantes por rango del puntaje global y tipo de colegio', value='tab-2'),
            dcc.Tab(label='Gráficas de número de estudiantes por ubicación y desempeño en el puntaje global', value='tab-3'),
            dcc.Tab(label='Gráfica Notas 1er y 2do Semestre', value='tab-4')
        ]
    ),
    html.Div(id='tab-content')
])

#Contenido de las pestañas
tab1_content = html.Div([
    html.P('Luego de completar algunos datos conocerás la probabilidad de una nota superior a 350 en las Pruebas Saber 11:', 
           style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    
    html.Div([
        html.Label('¿De qué año deseas conocer los resultados?'),
        dcc.Dropdown(id='Periodo', options=[{'label':'2020-1', 'value':1}, {'label':'2021-1', 'value':2}, {'label':'2022-1', 'value':3}, {'label':'2022-4', 'value':4}], placeholder='Periodo'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿En que zona esta ubicado el colegio?'),
        dcc.Dropdown(id='Ubicacion', options=[{'label':'Urbano', 'value':1}, {'label':'Rural', 'value':0}], placeholder='Ubicacion'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿El colegio es Bilingue?'),
        dcc.Dropdown(id='Bilingue', options=[{'label':'Si', 'value':1}, {'label':'No', 'value':0}], placeholder='Bilingue'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿Género del Estudiante?'),
        dcc.Dropdown(id='Genero', options=[{'label':'Femenino', 'value':0}, {'label':'Masculino', 'value':1}], placeholder='Genero'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Cuál es el estrato del estudiante?'),
        dcc.Dropdown(id='Estrato', options=[{'label':'Estrato 1', 'value':1}, {'label':'Estrato 2', 'value':2}, {'label':'Estrato 3', 'value':3}, {'label':'Estrato 4', 'value':4}, {'label':'Estrato 5', 'value':5}, {'label':'Estrato 6', 'value':6}], placeholder='Estrato'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿El estudiante tiene internet en la casa?'),
        dcc.Dropdown(id='Internet', options=[{'label':'No', 'value':0}, {'label':'Si', 'value':1}], placeholder='Internet'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuál rango cree usted que estaría la nota del estudiante en la sección de Ingles?'),
        dcc.Dropdown(id='Ingles', options=[{'label':'[0,20]', 'value':1}, {'label':'(20, 40]', 'value':2}, {'label':'(40, 60]', 'value':3}, {'label':'(60, 80]', 'value':4}, {'label':'(80, 100]', 'value':5}], placeholder='Ingles'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuá rango cree usted que estaría la nota del estudiante en la sección de Matematicas?'),
        dcc.Dropdown(id='Matematicas', options=[{'label':'[0,20]', 'value':1}, {'label':'(20, 40]', 'value':2}, {'label':'(40, 60]', 'value':3}, {'label':'(60, 80]', 'value':4}, {'label':'(80, 100]', 'value':5}], placeholder='Matematicas'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuá rango cree usted que estaría la nota del estudiante en la sección de Ciencias Sociales?'),
        dcc.Dropdown(id='Sociales', options=[{'label':'[0,20]', 'value':1}, {'label':'(20, 40]', 'value':2}, {'label':'(40, 60]', 'value':3}, {'label':'(60, 80]', 'value':4}, {'label':'(80, 100]', 'value':5}], placeholder='Sociales'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuá rango cree usted que estaría la nota del estudiante en la sección de Ciencias Naturales? '),
        dcc.Dropdown(id='Naturales', options=[{'label':'[0,20]', 'value':1}, {'label':'(20, 40]', 'value':2}, {'label':'(40, 60]', 'value':3}, {'label':'(60, 80]', 'value':4}, {'label':'(80, 100]', 'value':5}], placeholder='Naturales'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuá rango cree usted que estaría la nota del estudiante en la sección de Lectura Crítica? '),
        dcc.Dropdown(id='Lectura', options=[{'label':'[0,20]', 'value':1}, {'label':'(20, 40]', 'value':2}, {'label':'(40, 60]', 'value':3}, {'label':'(60, 80]', 'value':4}, {'label':'(80, 100]', 'value':5}], placeholder='Lectura'),

    ], className='row'),

    html.Br(),

    html.Button('Enviar datos', id='boton', n_clicks=0),

    html.Br(),

    html.Div(id='output')
])

tab2_content = html.Div([
    html.H2('Gráfica de Barras Número de Estudiantes vs Rango Puntaje Global'),
    html.P('A continuación, se muestra una gráfica de barras que representa el numero de estudiantes de acuerdo al tipo de colegio y su puntaje obtenido.'),
    grafica1()
])

tab3_content = html.Div([
    html.H2('Gráfica de Barras Número de Estudiantes vs Ubicación'),
    html.P('A continuación, se muestra una gráfica de barras que representa el numero de estudiantes de acuerdo al tipo de ubicaion y su puntaje obtenido. Es decir, si esta por encima o por debajo del puntaje promedio de admisión para universidades privadas del departamento.'),
    grafica2()
])

tab4_content = html.Div([
    html.H2('Gráfica de Barras Número de Estudiantes vs Servicio de Internet'),
    html.P('A continuación, se muestra una gráfica de barras que representa el numero de estudiantes de acuerdo al tipo de ubicaion y su puntaje obtenido.'),
    grafica3()
])

#Callbacks para actualizar el contenido de la pestaña que el usuario escoja
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    elif tab == 'tab-2':
        return tab2_content
    elif tab == 'tab-3':
        return tab3_content
    elif tab == 'tab-4':
        return tab4_content

# Definir tus callbacks aquí
@app.callback(
    Output('output', 'children'),
    [Input('boton', 'n_clicks')],
    [State('Periodo', 'value'),
    State('Ubicacion', 'value'),
    State('Bilingue', 'value'),
    State('Genero', 'value'),
    State('Estrato', 'value'),
    State('Internet', 'value'),
    State('Ingles', 'value'),
    State('Matematicas', 'value'),
    State('Sociales', 'value'),
    State('Naturales', 'value'),
    State('Lectura', 'value'),]
)
def predecir_probabilidad_obtener_puntaje_icfes(n_clicks, Periodo, Ubicacion, Bilingue, Genero , Estrato, Internet, Ingles, Matematicas, Sociales, Naturales, Lectura):
    if n_clicks > 0:
        ruta = "D:/datos usuario/Documents/Universidad de los Andes/2023-20/Analitica/Proyecto/Proyecto3/Basediscretizada.xlsx"
        datos = pd.read_excel(ruta)

        modelo = BayesianNetwork([('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'periodo'), ('cole_area_ubicacion', 'cole_bilingue'), ('estu_genero', 'punt_matematicas'), ('fami_tieneinternet', 'fami_estratovivienda'), ('fami_tieneinternet', 'cole_bilingue'), ('punt_ingles', 'punt_global'), ('punt_ingles', 'cole_area_ubicacion'), ('punt_ingles', 'fami_tieneinternet'), ('punt_ingles', 'fami_estratovivienda'), ('punt_matematicas', 'punt_c_naturales'), ('punt_matematicas', 'punt_lectura_critica'), ('punt_sociales_ciudadanas', 'punt_ingles'), ('punt_c_naturales', 'punt_sociales_ciudadanas'), ('punt_c_naturales', 'punt_lectura_critica'), ('punt_c_naturales', 'punt_global'), ('punt_lectura_critica', 'punt_sociales_ciudadanas'), ('punt_lectura_critica', 'punt_ingles')])
        X = datos.drop(columns=['punt_global'])
        y = datos['punt_global']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

        emv = MaximumLikelihoodEstimator(modelo, data=datos)
        modelo.fit(data=datos, estimator=MaximumLikelihoodEstimator)

        y_pred = modelo.predict(X_test)

        infer = VariableElimination(modelo)

        probabilidad_desercion = infer.query(["punt_global"], evidence={"periodo": Periodo , "cole_area_ubicacion": Ubicacion, "cole_bilingue": Bilingue, "estu_genero": Genero, "fami_estratovivienda": Estrato, "fami_tieneinternet": Internet, "punt_ingles": Ingles, "punt_matematicas": Matematicas, "punt_sociales_ciudadanas": Sociales, "punt_c_naturales": Naturales, "punt_lectura_critica": Lectura})

        return f'Probabilidad de sacar un buen puntaje para la universidad: {round(probabilidad_desercion.values[1] * 100, 2)}% y Probabilidad de sacar un mal puntaje para la universidad: {round(probabilidad_desercion.values[0] * 100, 2)}%'

if __name__ == '__main__':
    app.run_server(debug=True, port=8070)

