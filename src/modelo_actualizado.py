import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pickle


import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":

    data = pd.read_csv('./data/churn.csv')
   
    data = data.drop(data.columns[0:3], axis=1)


    # Convertimos los datos en formato categorico, para más info: shorturl.at/y0269
    column_equivalence = {}
    features = list(data.columns)
    for i, column in enumerate(list([str(d) for d in data.dtypes])):
        if column == "object":
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mode())
            categorical_column = data[data.columns[i]].astype("category")
            current_column_equivalence = dict(enumerate(categorical_column.cat.categories))
            column_equivalence[i] = dict((v,k) for k,v in current_column_equivalence.items())
            data[data.columns[i]] = categorical_column.cat.codes
        else:
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].median())

    print(data.head())
    print(column_equivalence)
            
    # Generar los datos para poder separar la variable de respuesta de los datos que tenemos disponibles
    X = data.copy()
    y = X.pop(data.columns[-1])


    # Separar los datos en datos de entrenamiento y testing
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    # Crear el modelo y entrenarlo
    clf_rf = RandomForestClassifier(random_state=0)
    clf_rf.fit(X_train, y_train)

    y_pred = clf_rf.predict(X_test)

    print(confusion_matrix(y_test, clf_rf.predict(X_test)))

    f1 = f1_score(y_test, y_pred)
   
    accuracy = accuracy_score(y_test, y_pred)

    print('F1-score:', f1)
    print('Accuracy:', accuracy)

    cm = confusion_matrix(y_test, y_pred)

    # Visualizar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, vmin=0, fmt='.0f', cbar=False, linewidths=.5, square=True, cmap='coolwarm')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('RandomForest')
    plt.savefig('figura_modelo_actualizado.png')
    plt.show()

    # Generar el binario del modelo para reutilizarlo, equivalencia de variables categoricas y caracteristicas del modelo
    """
    pickle.dump(clf_rf, open("churn/models/model.pk", "wb"))
    pickle.dump(column_equivalence, open("churn/models/column_equivalence.pk", "wb"))
    pickle.dump(features, open("churn/models/features.pk", "wb"))
    """

    
    """
    Balanceo de clases : Dado que el conjunto de datos parece estar desequilibrado, donde una clase es dominante sobre la otra, puedes intentar usar técnicas de balanceo de clases para igualar el número de muestras en ambas clases. La técnica que ya ha utilizado es el undersampling (RandomUnderSampler) para reducir la clase mayoritaria. Otra alternativa es el oversampling (RandomOverSampler) para aumentar la clase minoritaria o el uso de técnicas combinadas como SMOTE (Synthetic Minority Over-sampling Technique).
    """