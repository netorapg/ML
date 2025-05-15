import pandas as pd

dados = pd.read_csv('dados.csv')

distancia = dados['distancia']
desceu = dados['desceu']
classificacao = dados['classificacao']

print("Dataframe completo:")
print(dados)

print("\nDistancia:")
print(distancia)

print("\nDesceu:")
print(desceu)

print("\nClassificacao:")
print(classificacao)

ultima_linha = dados.iloc[-1]

print("\nUltima linha do arquivo:")
print(ultima_linha)

contagem_joao = (classificacao == 1).sum()
contagem_maria = (classificacao == 2).sum()

maior_distancia_joao = distancia[classificacao == 1].max()
maior_desceu_maria = desceu[classificacao == 2].max()

print("\n=== Resultados ===")
print(f"Total de linhas de João (classificação 1): {contagem_joao}")
print(f"Total de linhas de Maria (classificação 2): {contagem_maria}")
print(f"Maior distância de João: {maior_distancia_joao}")
print(f"Maior valor que Maria desceu: {maior_desceu_maria}")