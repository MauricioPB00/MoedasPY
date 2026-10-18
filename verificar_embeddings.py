
import json
import os

arquivo = "moedas_embeddings.json"

with open(arquivo, "r", encoding="utf-8") as f:
    dados = json.load(f)

total = len(dados)
com_frente = sum(1 for item in dados if item.get("embedding_obverse"))
com_verso = sum(1 for item in dados if item.get("embedding_reverse"))

print(f"Total de registros: {total:,}")
print(f"Com embedding da frente: {com_frente:,}")
print(f"Com embedding do verso: {com_verso:,}")
