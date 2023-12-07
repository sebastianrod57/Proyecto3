
import pandas as pd

def predecir_probabilidad_obtener_puntaje_icfes(Periodo, Ubicacion, Bilingue, Genero , Estrato, Internet, Ingles, Matematicas, Sociales, Naturales, Lectura):
    
    ruta = "/Users/sebastianrodriguez/Desktop/Proyecto3/nuevo_archivo.csv"

    datos = pd.read_csv(ruta)

    from pgmpy.models import BayesianNetwork
    modelo = BayesianNetwork ([('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'periodo'), ('cole_area_ubicacion', 'cole_bilingue'), ('estu_genero', 'punt_matematicas'), ('fami_tieneinternet', 'fami_estratovivienda'), ('fami_tieneinternet', 'cole_bilingue'), ('punt_ingles', 'punt_global'), ('punt_ingles', 'cole_area_ubicacion'), ('punt_ingles', 'fami_tieneinternet'), ('punt_ingles', 'fami_estratovivienda'), ('punt_matematicas', 'punt_c_naturales'), ('punt_matematicas', 'punt_lectura_critica'), ('punt_sociales_ciudadanas', 'punt_ingles'), ('punt_c_naturales', 'punt_sociales_ciudadanas'), ('punt_c_naturales', 'punt_lectura_critica'), ('punt_c_naturales', 'punt_global'), ('punt_lectura_critica', 'punt_sociales_ciudadanas'), ('punt_lectura_critica', 'punt_ingles')]
)

    from sklearn.model_selection import train_test_split
    # Dividir los datos
    # X crea un dataframe que contiene todas la columnas menos 'target'. Es decir, las caracteristicas que utilizara para hacer predicciones
    X = datos.drop(columns=['punt_global'])
    # Y crea una serie que contiene solo la columna "Target". 
    y = datos['punt_global']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    from pgmpy.estimators import MaximumLikelihoodEstimator
    emv = MaximumLikelihoodEstimator(modelo , data = datos)

    modelo.fit(data=datos , estimator = MaximumLikelihoodEstimator )

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    y_pred = modelo.predict(X_test)

    from pgmpy.inference import VariableElimination 
    infer = VariableElimination(modelo)

    # Utilizar el modelo Bayesian Network para predecir la probabilidad de deserción
    probabilidad_desercion = infer.query(["punt_global"], evidence={"periodo": Periodo , "cole_area_ubicacion": Ubicacion, "cole_bilingue": Bilingue, "estu_genero": Genero, "fami_estratovivienda": Estrato, "fami_tieneinternet": Internet, "punt_ingles": Ingles, "punt_matematicas": Matematicas, "punt_sociales_ciudadanas": Sociales, "punt_c_naturales": Naturales, "punt_lectura_critica": Lectura})

    return probabilidad_desercion

print(predecir_probabilidad_obtener_puntaje_icfes(4, 0, 0, 0, 2, 1, 3, 3, 3,3,3))
print(predecir_probabilidad_obtener_puntaje_icfes(4, 1, 0, 1, 5, 1, 4, 4, 4,4,5))
print(predecir_probabilidad_obtener_puntaje_icfes(4, 1, 1, 1, 5, 1, 4, 4, 4,4,5))
print(predecir_probabilidad_obtener_puntaje_icfes(4, 1, 0, 0, 4, 1, 4, 4, 4,4,4))
print(predecir_probabilidad_obtener_puntaje_icfes(4, 1, 0, 1, 5, 1, 4, 4, 5,5,5))