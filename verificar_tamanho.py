
import os

arquivo = r"C:\Users\Usuario\Documents\Moedas\MoedasPY\moedas_embeddings.json"

if os.path.exists(arquivo):
    tamanho = os.path.getsize(arquivo) / (1024 ** 2)
    print(f"Tamanho: {tamanho:.2f} MiB")
else:
    print("Arquivo não encontrado:", arquivo)
