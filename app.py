from math import e
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif

#cargamos y mostramos el logo en la barra lateral

logo = "imagen.png"
st.sidebar.image(logo, width=300)


# Función para cargar el conjunto de datos desde un archivo CSV
def cargar_datos_desde_csv(ruta_archivo):
    return pd.read_csv(ruta_archivo)

# Función para realizar undersampling y devolver los datos balanceados
def realizar_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

# Función para convertir los códigos de materia a nombres de materia
def convertir_codigo_a_materia(codigo_materia):
    # Agrega tu lógica de conversión aquí
    # Por ejemplo, puedes tener un diccionario que mapee códigos a nombres
    codigo_a_nombre = {0: 'EMPRENDIMIENTO',
                       1: 'GESTION DE PROYECTOS', 
                       2: 'COMPUTACIÓN Y SOCIEDAD', 
                       3: 'FUNDAMENTOS DE TECNOLOGIAS', 
                       4: 'ORGANIZACION Y ADMIN DE INFRE',
                       5:'FUNDAMENTOS DE BASE DE DATOS',
                       6:'COMUNICACION DE DATOS',
                       7:'ADMINISTRACION DE BASE DE DATOS',
                       8:'REDES DE DISPOSITIVOS',
                       9:'ARQUITECTURA DE REDES',
                       10:'SISTEMAS DISTRIBUIDOS',
                       11:'ALGORITMOS Y RESOLUCION DE PRO',
                       12:'FUNDAMENTOS DE HADWARE',
                       13:'COMPUTACION UBICUA',
                       14:'FUNDAMENTOS DE PROGRAMACION',
                       15:'ESTRUCTURAS DISCRETAS',
                       16:'ARQUITECTURA DE COMPUTADORAS',
                       17:'PROGRAMACION ORIENTADA A OBJET',
                       18:'ESTRUCTURA DE DATOS',
                       19:'COMUNICACION TECNICA Y PROFESIONAL',
                       20:'TECNOLOGIAS WEB',
                       21:'MODELADO DE SISTEMAS',
                       22:'FUNDAMENTOS DE INTERACCION HUM',
                       23:'DESARROLLO WEB',
                       24:'PLANIFICACION ESTRATEGICA Y SI',
                       25:'METODOLOGIA DE DESARROLLO',
                       26:'GESTION DE LA CALIDAD DEL SOFT',
                       27:'INGENIERIA DE REQUISITOS',
                       28:'PLATAFORMAS EMERGENTES',
                       29:'DESARROLLO BASADO EN PLAT WEB',
                       30:'ARQUITECTURA DE SOFTWARE',
                       31:'ARQUITECTURA EMPRESARIAL',
                       32:'MODELADO DE USUARIO',
                       33:'DESARROLLO BASADO EN PLATAF MO',
                       34:'FUNDAMENTOS Y APLICACION DE SE',
                       35:'PRACTICUM 4.1',
                       36:'APLICACION DE INTERA HUMA COMP',
                       37:'PROGRAMACION INTEGRATIVA',
                       38:'GOBERNANZA DE TECNOL DE INFOR',
                       39:'PROYECTOS DE TECNOLO DE INFORM',
                       40:'METODOLOGIA DE LA INVEST Y TEC',
                       41:'ESTADISTICA PARA LAS ING Y ARQ',
                       42:'APLICACION DE MATEMATICAS Y ES',
                       43:'ETICA Y MORAL',
                       44:'COMPOSICION DE TEXTOS CIENTIFI',
                       45:'FUNDAMENTOS MATEMATICOS',
                       46:'ALGEBRA LINEAL',
                       47:'CALCULO DIFERENCIAL',
                       48:'CALCULO INTEGRAL',
                       49:'HUMANISMO, UNIVERSIDAD Y CULTURA',
                       50:'ANTROPOLOGIA BASICA',
                       51:'FUNDAMENTOS DE INGENIERIA DE S',
                       52:'EVALUCION DE LA SEGUR EN SIST',
                       }
    return codigo_a_nombre.get(codigo_materia, f'Materia Desconocida ({codigo_materia})')

# Función para convertir los códigos de materia a nombres de materia
def convertir_codigo_a_sexo(codigo_sexo):
    # Agrega tu lógica de conversión aquí
    # Por ejemplo, puedes tener un diccionario que mapee códigos a nombres
    codigo_a_nom = {0: 'HOMBRE',
                       1: 'MUJER', 
                       2: 'SIN ESPECIFICAR', 
                       }
    return codigo_a_nom.get(codigo_sexo, f'Sexo Desconocido ({codigo_sexo})')

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train, modelo_seleccionado):
#        # Seleccionar modelo y entrenar
    if modelo_seleccionado == "K-NN":
        k=3
        modelo = KNeighborsClassifier(n_neighbors=k)
    elif modelo_seleccionado == "Naive Bayes":
        modelo = GaussianNB()
    elif modelo_seleccionado == "Random Forest":
        modelo = RandomForestClassifier()
    elif modelo_seleccionado == "Decision Tree":
        modelo = DecisionTreeClassifier()
    elif modelo_seleccionado == "Logistic Regression":
        modelo = LogisticRegression()
    else:
        st.error("Algoritmo no reconocido.")

    modelo.fit(X_train, y_train)
    return modelo

# Función para realizar predicciones y mostrar resultados
def mostrar_predicciones(modelo, X_test, y_test, materia_seleccionada, ciclo_seleccionado):
    # Filtrar los datos por materia y ciclo seleccionados
    datos_filtrados = X_test[(X_test['Curso'] == materia_seleccionada) & (X_test['Sexo'] == ciclo_seleccionado)]
    y_test_filtrado = y_test.loc[datos_filtrados.index]

    predicciones = modelo.predict(datos_filtrados)
    exactitud = accuracy_score(y_test_filtrado, predicciones)

    st.subheader(f"Resultados para {materia_seleccionada} - Ciclo {ciclo_seleccionado}")
    st.write("Predicciones realizadas por el modelo:")
    st.write(predicciones)

    st.write("Exactitud del modelo en los datos filtrados:")
    st.write(f"{exactitud * 100:.2f}%")

    # Mostrar la matriz de confusión
    matriz_confusion = confusion_matrix(y_test_filtrado, predicciones)
    st.subheader("Matriz de Confusión")
    st.write(matriz_confusion)

    # Mostrar el reporte de clasificación
    reporte_clasificacion = classification_report(y_test_filtrado, predicciones)
    st.subheader("Reporte de Clasificación")
    st.write(reporte_clasificacion)


# Función para mostrar la importancia de las características en Streamlit
#def mostrar_importancia_caracteristicas(modelo, X, X_train, y_train):
    
#    return entrenar_modelo
    

# Función principal de la aplicación
def main():
    st.title("Modelo para predecir el desempeño academico")

    # Cargar datos desde el archivo CSV
    ruta_archivo = "datosFinales.csv"  # Reemplaza con el nombre de tu archivo CSV
    datos = cargar_datos_desde_csv(ruta_archivo)

    # Transformar el código de materia a nombre de materia
    datos['Curso'] = datos['Curso'].apply(convertir_codigo_a_materia)
    datos['Sexo'] = datos['Sexo'].apply(convertir_codigo_a_sexo)

    # Separar características (X) y variable objetivo (y)
    X = datos.drop(columns=["Status","Curso","Curso","Sexo","Sexo"])
    y = datos["Status"]
    
    # Balancear datos con undersampling
    X_resampled, y_resampled = realizar_undersampling(X, y)
    
    # Menú de navegación para seleccionar el modelo
    modelo_seleccionado = st.sidebar.selectbox("Seleccione el Modelo", ["Random Forest", "Decision Tree", "K-NN", "Naive Bayes","Logistic Regression"])    
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train, modelo_seleccionado)
    
    # Selector de materia y ciclo
    materias = sorted(datos['Curso'].unique())
    materia_seleccionada = st.selectbox("Curso",["Todos los cursos"]+list(materias))
    genero = sorted(datos['Sexo'].unique())
    genero_seleccionado = st.selectbox("Seleccione el Sexo", ["Todos los generos"]+list(genero))

    #Filtrar los datos según la materia y el ciclo seleccionados
    datos_filtrados = datos[(datos['Curso'] == materia_seleccionada) & (datos['Sexo'] == genero_seleccionado)]

    # Filtrar los datos según la materia y el género seleccionados
    if materia_seleccionada == "Todos los cursos" and genero_seleccionado == "Todos los generos":
        datos_filtrados = datos
    elif materia_seleccionada != "Todos los cursos" and genero_seleccionado == "Todos los generos":
        datos_filtrados = datos[datos['Curso'] == materia_seleccionada]
    elif materia_seleccionada == "Todos los cursos" and genero_seleccionado != "Todos los generos":
        datos_filtrados = datos[datos['Sexo'] == genero_seleccionado]
    else:
        datos_filtrados = datos[(datos['Curso'] == materia_seleccionada) & (datos['Sexo'] == genero_seleccionado)]
    
    # Mostrar resumen total si no se ha seleccionado ninguna materia o género específico
    if materia_seleccionada == "Todos los cursos" and genero_seleccionado == "Todos los generos":
        st.subheader("Sumatoria Total")
        total_aprobados = datos_filtrados[datos_filtrados['Status'] == 0].shape[0]
        total_reprobados = datos_filtrados[datos_filtrados['Status'] == 1].shape[0]
        st.write("Número total de Aprobados:", total_aprobados)
        st.write("Número total de Reprobados:", total_reprobados)
    else:
        st.subheader(f"Resultados para {materia_seleccionada if materia_seleccionada != 'Todos los cursos' else 'Todos los cursos'} - Género {genero_seleccionado if genero_seleccionado != 'Todos los generos' else 'Todos los generos'}")
        st.write("Número de Aprobados:", datos_filtrados[datos_filtrados['Status'] == 1].shape[0])
        st.write("Número de Reprobados:", datos_filtrados[datos_filtrados['Status'] == 0].shape[0])
    
    # Mostrar importancia de características
    st.subheader("Importancia de Características")
    if modelo_seleccionado == "K-NN" or modelo_seleccionado == "Naive Bayes":
        feature_importance = mutual_info_classif(X_train, y_train)
        st.bar_chart(pd.Series(feature_importance, index=X.columns).sort_values(ascending=False))
    elif modelo_seleccionado == "Logistic Regression":
        st.bar_chart(pd.Series(modelo.coef_[0], index=X.columns).sort_values(ascending=False))
    elif modelo_seleccionado == "Random Forest" or modelo_seleccionado == "Decision Tree":
        importancias = modelo.feature_importances_
        importancia_df = pd.DataFrame({"Característica": X.columns, "Importancia": importancias})
        importancia_df = importancia_df.sort_values(by="Importancia", ascending=False)
        st.bar_chart(importancia_df.set_index("Característica"))
if __name__ == "__main__":
    main()
