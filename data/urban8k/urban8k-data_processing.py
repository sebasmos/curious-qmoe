#!/usr/bin/env python
# QWave embedding extractor – UrbanSound8K

import os, argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision.models import efficientnet_b3
from PIL import Image
import clip  # ViT family

# ----------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, metadata_csv, image_root_dir, transform):
        self.df = pd.read_csv(metadata_csv)
        self.image_root_dir = image_root_dir
        self.transform = transform

        # Update filenames to match .png instead of original .wav
        self.df['image_name'] = self.df['slice_file_name'].str.replace('.wav', '.png', regex=False)

        # Check available images
        self.df['full_path'] = self.df.apply(
            lambda row: os.path.join(image_root_dir, f"fold{row['fold']}", row['image_name']), axis=1
        )
        self.df = self.df[self.df['full_path'].apply(os.path.exists)]

        if self.df.empty:
            raise RuntimeError(f"No valid PNG files found in {image_root_dir}")
        print(f"Found {len(self.df)} images in {image_root_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['full_path']).convert("RGB")
        return self.transform(img), int(row['classID']), os.path.basename(row['full_path'])

# ----------------------------------------------------------------------
def extract_and_save_embeddings(model, loader, device, out_dir, metadata_df):
    meta_rows, all_embs = [], []

    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            embs = model.encode_image(imgs) if hasattr(model, "encode_image") else model(imgs)
            all_embs.append(embs.cpu().numpy())

            for lbl, nm in zip(labels.numpy(), names):
                meta_rows.append([nm, int(lbl)])

    # Embeddings → DataFrame
    emb_df = pd.DataFrame(np.vstack(all_embs))
    meta_df = pd.DataFrame(meta_rows, columns=["name", "label"])

    # Unir con metadata original (usa slice_file_name para empatar)
    merged = pd.merge(meta_df, metadata_df, left_on="name", right_on="slice_file_name")

    # Mapeo real: classID → class name
    class_to_category = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }

    merged["category"] = merged["label"].map(class_to_category)
    merged["class_id"] = merged["category"].astype("category").cat.codes.astype(np.int8)
    merged = merged.rename(columns={"fold": "folder"})

    # Mantener columnas necesarias
    merged = merged[["folder", "name", "label", "category", "class_id"]].reset_index(drop=True)

    # Concatenar metadata + embeddings
    final_df = pd.concat([merged, emb_df], axis=1)

    # Guardar
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "urbansound8k.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved ➜ {csv_path}  shape={final_df.shape}")

# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("QWave embedding extractor – UrbanSound8K")
    p.add_argument("--data_dir",   default="/home/ltorres/QWave/archive")
    p.add_argument("--output_dir", default="/home/ltorres/QWave/Results")
    p.add_argument("--meta_csv",   default="/home/ltorres/QWave/archive/UrbanSound8K.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--embedding_size", type=int, default=None)
    p.add_argument("--model_type", type=str, default="efficientnet",
                   choices=["efficientnet", "vit-b/16", "vit-b/32", "vit-l/14", "vit-l/14@336px"],
                   help="Model type to use for embedding extraction")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = args.data_dir

    # --- load full metadata
    meta_df = pd.read_csv(args.meta_csv)

    # --- modelo por argumento
    model_type = args.model_type
    label = model_type.replace("/", "_").replace("@", "_at_")
    model_out_dir = os.path.join(args.output_dir, label)

    # cargar modelo
    if model_type == "efficientnet":
        model = efficientnet_b3(weights="IMAGENET1K_V1")
        in_dim  = model.classifier[1].in_features
        out_dim = args.embedding_size or in_dim
        model.classifier[1] = nn.Linear(in_dim, out_dim)
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])
    else:
        clip_map = {
            "vit-b/16":"ViT-B/16", "vit-b/32":"ViT-B/32",
            "vit-l/14":"ViT-L/14", "vit-l/14@336px":"ViT-L/14@336px"
        }
        model, transform = clip.load(clip_map[model_type], device=device)

    model.eval().to(device)

    # ✅ Llamada corregida
    ds = CustomImageDataset(args.meta_csv, image_dir, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Embeddings
    extract_and_save_embeddings(model, dl, device, model_out_dir, meta_df)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
