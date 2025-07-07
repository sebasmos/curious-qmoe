#!/usr/bin/env python
# QWave – ABGQI mel-spectrogram embedding extractor (PyTorch + CLIP + EfficientNetB3)

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch import nn
import clip

# ----------------------------------------------------------------------
class UP3ImageDataset(Dataset):
    """Recorre subcarpetas y retorna imágenes, clases y nombres."""
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(class_dir, fname), class_name))

        if not self.samples:
            raise RuntimeError(f"No se encontraron imágenes en {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, cls, os.path.basename(path)

# ----------------------------------------------------------------------
def extract_embeddings(model, dataloader, device, out_dir):
    model.eval()
    meta_rows, all_embs = [], []

    with torch.no_grad():
        for imgs, classes, names in tqdm(dataloader, desc="Extrayendo embeddings"):
            imgs = imgs.to(device)
            # Para CLIP, usar encode_image; para EfficientNet directo
            if hasattr(model, "encode_image"):
                embs = model.encode_image(imgs)
            else:
                embs = model(imgs)
            embs = embs.cpu().numpy()
            all_embs.append(embs)
            for cls, name in zip(classes, names):
                meta_rows.append([cls, name])

    emb_arr = np.vstack(all_embs)
    emb_df = pd.DataFrame(emb_arr)
    meta_df = pd.DataFrame(meta_rows, columns=["category", "name"])
    meta_df["class_id"] = meta_df["category"].astype("category").cat.codes.astype(np.int8)
    final_df = pd.concat([meta_df, emb_df], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ABGQI_embeddings_torch.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved ➜ {csv_path}  shape={final_df.shape}")

# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("QWave ABGQI embedding extractor (PyTorch + CLIP + EfficientNetB3)")
    p.add_argument("--data_dir", default=r"C:\Users\cadur\OneDrive\Documentos\Plantillas personalizadas de Office\QWave-dev\data\ABGQI_mel_spectrograms",
                   help="Ruta a carpeta ABGQI con subcarpetas por categoría")
    p.add_argument("--output_dir", default=r"C:\Users\cadur\OneDrive\Documentos\Plantillas personalizadas de Office\QWave-dev\Result\Quinn",
                   help="Directorio para guardar los CSV")
    p.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote para DataLoader")
    p.add_argument("--cpu", action="store_true", help="Forzar uso de CPU incluso si hay GPU disponible")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model_list = ["efficientnet", "vit-b/16", "vit-b/32", "vit-l/14", "vit-l/14@336px"]
    clip_map = {
        "vit-b/16": "ViT-B/16",
        "vit-b/32": "ViT-B/32",
        "vit-l/14": "ViT-L/14",
        "vit-l/14@336px": "ViT-L/14@336px"
    }

    for model_type in model_list:
        print(f"\n==== Procesando modelo: {model_type} ====")
        label = model_type.replace("/", "_").replace("@", "_at_")

        if model_type == "efficientnet":
            base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            base.classifier = nn.Identity()
            model = base.to(device)
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            model, transform = clip.load(clip_map[model_type], device=device)

        dataset = UP3ImageDataset(args.data_dir, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        out_dir = os.path.join(args.output_dir, f"{label}_ABGQI")
        extract_embeddings(model, loader, device, out_dir)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()