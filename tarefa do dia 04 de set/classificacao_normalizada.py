import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import warnings

def carregar_e_preprocessar_dados():
    print("Carregando dataset...")
    df = pd.read_csv('adult.csv')
    
    # Remover espaços em branco dos dados string
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Converter strings para inteiros usando LabelEncoder
    le = LabelEncoder()
    colunas_categoricas = df.select_dtypes(include=['object']).columns
    
    print("\nConvertendo colunas categóricas:")
    for coluna in colunas_categoricas:
        print(f"- Convertendo: {coluna}")
        df[coluna] = le.fit_transform(df[coluna])
    
    # Normalizar dados numéricos usando MinMaxScaler
    scaler = MinMaxScaler()
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    
    print("\nNormalizando colunas numéricas:")
    for coluna in colunas_numericas:
        print(f"- Normalizando: {coluna}")
        df[coluna] = scaler.fit_transform(df[coluna].values.reshape(-1, 1))
    
    # Separar features e target
    X = df.drop('class', axis=1)
    y = df['class']
    
    return X, y

def treinar_e_avaliar_modelos(X, y):
    # Definir modelos
    modelos = {
        'Perceptron': Perceptron(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    resultados = {}
    
    print("\nTreinando e avaliando modelos:")
    print("="*50)
    
    for nome, modelo in modelos.items():
        print(f"\nModelo: {nome}")
        print("-"*20)
        
        # Treinar modelo
        modelo.fit(X_train, y_train)
        
        # Avaliar com cross-validation
        scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
        
        # Fazer predições no conjunto de teste
        y_pred = modelo.predict(X_test)
        
        # Guardar resultados
        resultados[nome] = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        print("\nClassification Report:")
        print(resultados[nome]['classification_report'])
    
    return resultados

def main():
    print("CLASSIFICAÇÃO COM DADOS NORMALIZADOS")
    print("="*50)
    
    try:
        # Carregar e preprocessar dados
        X, y = carregar_e_preprocessar_dados()
        
        # Treinar e avaliar modelos
        resultados = treinar_e_avaliar_modelos(X, y)
        
        # Mostrar ranking final
        print("\nRANKING FINAL DOS MODELOS")
        print("="*50)
        
        # Ordenar modelos por accuracy média do CV
        ranking = sorted(
            [(nome, res['cv_mean']) for nome, res in resultados.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (nome, score) in enumerate(ranking, 1):
            print(f"{i}º {nome}: {score:.4f}")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()