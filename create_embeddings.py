import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

IMAGES_DIR = r"C:\Users\Usuario\Documents\Moedas\Moedas\src\assets\img\imagens"

OUTPUT_FILE = "moedas_embeddings.json"

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando dispositivo:", device)

    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def generate_embedding(model, preprocess, device, image_path):
    """Gera um embedding de uma imagem usando CLIP."""
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print("Erro ao carregar imagem:", image_path)
        return None

    image_preprocessed = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_preprocessed)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()


def main():
    model, preprocess, device = load_clip_model()

    database = []

    files = os.listdir(IMAGES_DIR)
    obverses = [f for f in files if "_obverse" in f]
    reverses = [f for f in files if "_reverse" in f]

    print("\nGerando embeddings...\n")

    for obv in tqdm(obverses):
        id_str = obv.split("_")[0]

        obv_path = os.path.join(IMAGES_DIR, obv)
        rev_path = os.path.join(IMAGES_DIR, f"{id_str}_reverse.jpg")

        entry = {"id": int(id_str)}

        entry["embedding_obverse"] = generate_embedding(
            model, preprocess, device, obv_path
        )

        if os.path.exists(rev_path):
            entry["embedding_reverse"] = generate_embedding(
                model, preprocess, device, rev_path
            )
        else:
            entry["embedding_reverse"] = None

        database.append(entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)

    print("\n✅ Embeddings gerados com sucesso!")
    print(f"Arquivo salvo como: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
