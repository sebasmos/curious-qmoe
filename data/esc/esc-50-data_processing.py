#!/usr/bin/env python
# QWave – ESC-50 mel-spectrogram embedding extractor

import os, argparse, numpy as np, pandas as pd, torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision.models import efficientnet_b3
from PIL import Image
import clip   # ViT family
# ----------------------------------------------------------------------
class CustomImageDataset(Dataset):
    """
    PNG names: FOLD-CLIPID-TAKE-TARGET.png  e.g. 1-100210-A-36.png
    """
    def __init__(self, image_dir: str, transform):
        self.image_dir   = image_dir
        self.transform   = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        if not self.image_files:
            raise RuntimeError(f"No PNG files found in {image_dir}")
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):  return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]                        # full file name
        stem  = fname[:-4]                                   # drop .png
        parts = stem.split('-')                              # [fold, clip, take, target]
        if len(parts) != 4 or not parts[0].isdigit():
            raise ValueError(f"Bad filename pattern: {fname}")
        label = int(parts[3])                                # TARGET as int
        path  = os.path.join(self.image_dir, fname)
        img   = Image.open(path).convert("RGB")
        return self.transform(img), label, fname             # keep fname for later
# ----------------------------------------------------------------------
def extract_and_save_embeddings(model, loader, device, out_dir):
    meta_rows, all_embs = [], []

    with torch.no_grad():
        for imgs, labels, names in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            embs = model.encode_image(imgs) if hasattr(model, "encode_image") else model(imgs)
            all_embs.append(embs.cpu().numpy())

            for lbl, nm in zip(labels.numpy(), names):
                fold, clip_id, take, target = nm[:-4].split('-')
                meta_rows.append([int(fold), nm, int(target)])

    # embeddings -> DataFrame
    emb_df = pd.DataFrame(np.vstack(all_embs))
    meta_df = pd.DataFrame(meta_rows, columns=["folder", "name", "label"])

    # 5-class mapping  -----------------------------------------------
    class_to_category = { 0:"Animals",1:"Animals",2:"Animals",3:"Animals",4:"Animals",
        5:"Natural soundscapes",6:"Natural soundscapes",7:"Natural soundscapes",
        8:"Natural soundscapes",9:"Natural soundscapes",10:"Human non-speech",
        11:"Human non-speech",12:"Human non-speech",13:"Human non-speech",
        14:"Human non-speech",15:"Interior sounds",16:"Interior sounds",
        17:"Interior sounds",18:"Interior sounds",19:"Interior sounds",
        20:"Exterior noises",21:"Exterior noises",22:"Exterior noises",
        23:"Exterior noises",24:"Exterior noises",25:"Animals",26:"Animals",
        27:"Animals",28:"Animals",29:"Animals",30:"Natural soundscapes",
        31:"Natural soundscapes",32:"Natural soundscapes",33:"Natural soundscapes",
        34:"Natural soundscapes",35:"Human non-speech",36:"Human non-speech",
        37:"Human non-speech",38:"Human non-speech",39:"Human non-speech",
        40:"Interior sounds",41:"Interior sounds",42:"Interior sounds",
        43:"Interior sounds",44:"Interior sounds",45:"Exterior noises",
        46:"Exterior noises",47:"Exterior noises",48:"Exterior noises",
        49:"Exterior noises" }
    meta_df["category"] = meta_df["label"].map(class_to_category)
    meta_df["class_id"] = meta_df["category"].astype("category").cat.codes.astype(np.int8)

    # concat so meta cols come first
    final_df = pd.concat([meta_df, emb_df], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "esc-50.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved ➜ {csv_path}  shape={final_df.shape}")
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("QWave embedding extractor")
    p.add_argument("--data_dir",   default="/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/Mels_folds_dataset") 
    p.add_argument("--output_dir", default="/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_type", choices=["efficientnet","vit-b/16","vit-b/32","vit-l/14","vit-l/14@336px"],
                   default="efficientnet")
    p.add_argument("--embedding_size", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label  = args.model_type.replace("/","_").replace("@","_at_")

    # -------- model & transform
    if args.model_type == "efficientnet":
        model = efficientnet_b3(weights="IMAGENET1K_V1")
        in_dim = model.classifier[1].in_features
        out_dim = args.embedding_size or in_dim
        model.classifier[1] = nn.Linear(in_dim, out_dim)
        embed_dim = out_dim
        transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor()])
    else:
        clip_map = {"vit-b/16":"ViT-B/16", "vit-b/32":"ViT-B/32",
                    "vit-l/14":"ViT-L/14", "vit-l/14@336px":"ViT-L/14@336px"}
        model, transform = clip.load(clip_map[args.model_type], device=device)
        embed_dim = model.visual.output_dim

    model.eval().to(device)

    out_dir = os.path.join(args.output_dir, f"{label}_{embed_dim}")
    ds  = CustomImageDataset(args.data_dir, transform)
    dl  = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    extract_and_save_embeddings(model, dl, device, out_dir)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()