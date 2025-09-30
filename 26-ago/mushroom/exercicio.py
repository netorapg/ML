"""
===============================================================================
EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS - DATASET MUSHROOM
===============================================================================
Requisitos:
1. Fazer o código funcionar com dataset mushroom (cogumelos)
2. Dividir dataset em 80% treino / 20% teste
3. Calcular F1-Score para cada algoritmo
4. Acrescentar resultados em listas
5. Repetir 20 vezes para cada algoritmo
6. Calcular média e desvio padrão para cada algoritmo
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from random import shuffle
import warnings


def carregar_dataset_mushroom():
    print("Carregando dataset mushroom...")
    
    # Definir nomes das colunas do dataset de cogumelos
    column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
        'stalk-surface-below-ring', 'stalk-color-above-ring', 
        'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 
        'population', 'habitat'
    ]
    
    df = pd.read_csv('agaricus-lepiota.data', 
                     names=column_names, na_values='?')
    
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


def processar_dataset(df):
    print("\n Processando dataset...")
    
    linhas_antes = len(df)
    df_clean = df.dropna()
    linhas_depois = len(df_clean)
    
    print(f"   Dados faltantes removidos: {linhas_antes - linhas_depois} linhas")
    print(f"   Dataset final: {linhas_depois} amostras")
    
    df_processed = df_clean.copy()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    print(f"   Convertendo {len(categorical_columns)} colunas categóricas...")
    
    le = LabelEncoder()
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    print("Dataset processado!")
    return df_processed


def preparar_dados_ml(df, train_ratio=0.8):
    print(f"\nPreparando dados para ML (Treino: {train_ratio*100:.0f}% / Teste: {(1-train_ratio)*100:.0f}%)...")
    
    # Para o dataset de cogumelos, o target é a coluna 'class'
    target_column = 'class'
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    indices = list(range(len(y)))
    shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    split_point = int(len(y) * train_ratio)
    
    X_train = X_shuffled[:split_point]
    y_train = y_shuffled[:split_point]
    X_test = X_shuffled[split_point:]
    y_test = y_shuffled[split_point:]
    
    print(f"Dados preparados: {len(X_train)} treino, {len(X_test)} teste")
    print(f"Classes únicas: {np.unique(y)}")
    
    return X_train, X_test, y_train, y_test


def executar_experimentos(X_train, X_test, y_train, y_test, num_execucoes=20):
    print(f"\nEXECUTANDO EXPERIMENTOS - {num_execucoes} EXECUÇÕES")
    print("="*70)
    
    print("Normalizando dados...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rng = np.random.RandomState()
    algoritmos = {
        'Perceptron': lambda: Perceptron(max_iter=1000, random_state=rng),
        'SVM': lambda: SVC(probability=True, gamma='auto', random_state=rng),
        'Naive Bayes': lambda: GaussianNB(),
        'Decision Tree': lambda: DecisionTreeClassifier(random_state=rng, max_depth=10),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=7)
    }
    
    resultados = {nome: [] for nome in algoritmos.keys()}
    
    warnings.filterwarnings('ignore')
    
    print("\nIniciando execuções...")
    print("Exec | " + " | ".join([f"{nome:>12}" for nome in algoritmos.keys()]))
    print("-" * (6 + 15 * len(algoritmos)))
    
    for execucao in range(num_execucoes):
        scores_execucao = []
        
        for nome, criar_modelo in algoritmos.items():
            modelo = criar_modelo()
            modelo.fit(X_train_scaled, y_train)
            
            y_pred = modelo.predict(X_test_scaled)
            
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            resultados[nome].append(f1)
            scores_execucao.append(f1)
        
        scores_str = " | ".join([f"{score:>12.4f}" for score in scores_execucao])
        print(f"{execucao+1:3d}  | {scores_str}")
    
    warnings.filterwarnings('default')
    
    return resultados

def calcular_estatisticas(resultados):
    print(f"\nESTATÍSTICAS FINAIS")
    print("="*80)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<15} | {'Média':<8} | {'Desvio':<8} | {'Mínimo':<8} | {'Máximo':<8}")
    print("-" * 65)
    
    for nome, scores in resultados.items():
        media = np.mean(scores)
        desvio = np.std(scores)
        minimo = np.min(scores)
        maximo = np.max(scores)
        
        estatisticas[nome] = {
            'media': media,
            'desvio': desvio,
            'minimo': minimo,
            'maximo': maximo,
            'scores': scores
        }
        
        print(f"{nome:<15} | {media:<8.4f} | {desvio:<8.4f} | {minimo:<8.4f} | {maximo:<8.4f}")
    
    return estatisticas


def exibir_ranking(estatisticas):
    print(f"\nRANKING DE PERFORMANCE (por F1-Score médio)")
    print("="*50)
    
    # Ordenar por média decrescente
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media'], reverse=True)
    
    medalhas = ["1º", "2º", "3º", "4º", "5º"]
    
    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"{i+1}º"
        print(f"{medalha} {nome:<15}: {stats['media']:.4f} (±{stats['desvio']:.4f})")



def executar_experimentos_cv(X, y, num_execucoes=20, cv_folds=5):
    print(f"\nEXECUTANDO EXPERIMENTOS COM CROSS-VALIDATOIN")
    print(f"Folds: {cv_folds} | Execução: {num_execucoes}")
    print("="*70)

    print("Normalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    algoritmos = {
        'Perceptron': Perceptron(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, gamma='auto', random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'KNN': KNeighborsClassifier(n_neighbors=7)
    }
    
    resultados = {nome: [] for nome in algoritmos.keys()}
    
    warnings.filterwarnings('ignore')
    
    print("\nIniciando execuções...")
    print("Exec | " + " | ".join([f"{nome:>12}" for nome in algoritmos.keys()]))
    print("-" * (6 + 15 * len(algoritmos)))
    
    for execucao in range(num_execucoes):
        scores_execucao = []
        
        cv_execucao - StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42 + execucao)
        
        for nome, modelo in algoritmos.itmes():
            cv_scores = cross_val_score(modelo, X_scaled, y, cv=cv_execucao, scoring='f1_macro', n_jobs=-1)
            
            f1_medio = np.mean(cv_scores)
            
            resultados[nome].append(f1_medio)
            scores_execucao.append(f1_medio)
        
        scores_str = " | ".join([f"{score:>12.4f}" for score in scores_execucao])
        print(f"{execucao+1:3d} | {scores_str}")

    warnings.filterwarnings('default')

    return resultados


def preparar_dados_cv(df):
    """Prepara dados para cross-validation (sem divisão treino/teste)"""
    print(f"\nPreparando dados para Cross-Validation...")
    
    target_column = 'class'
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    print(f"Dados preparados: {len(X)} amostras totais")
    print(f"Classes únicas: {np.unique(y)}")

    return X, y

def main():
    print("EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS ML COM CROSS-VALIDATION")
    print("="*70)
    
    try:
        df = carregar_dataset_mushroom()
        
        df_processed = processar_dataset(df)
        
        #X_train, X_test, y_train, y_test = preparar_dados_ml(df_processed, train_ratio=0.8)
        X, y = preparar_dados_cv(df_processed)

        resultados = executar_experimentos_cv(X, y, num_execucoes=20)

        estatisticas = calcular_estatisticas(resultados)
        
        exibir_ranking(estatisticas)
        
        print(f"\nEXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f"ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

