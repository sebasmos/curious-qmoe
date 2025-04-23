import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from pathlib import Path
import gc


class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "ESC-50-master.zip"
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta', 'esc50.csv'),
    }

    def __init__(self, root, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.mel_spectrograms_dir = os.path.join(root, 'ESC-50-master', 'Mels_folds_dataset')
        os.makedirs(self.mel_spectrograms_dir, exist_ok=True)

        print("Loading audio files...")
        self.df['category'] = self.df['category'].str.replace('_', ' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

        print(f"Audio files are saved in: {os.path.join(self.root, self.base_folder, self.audio_dir)}")
        print(f"Mel spectrograms will be saved in: {self.mel_spectrograms_dir}")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_', ' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def saveMel(self, signal, sr, mel_save_path):
        """Save a mel spectrogram from an audio signal."""
        N_FFT = 2048
        HOP_SIZE = 32
        N_MELS = 128
        WIN_SIZE = 2048
        WINDOW_TYPE = 'hann'
        FMIN = 0
        FMAX = 20000

        S = librosa.feature.melspectrogram(
            y=signal, 
            sr=sr, 
            hop_length=HOP_SIZE, 
            n_fft=N_FFT, 
            n_mels=N_MELS, 
            fmin=FMIN, 
            fmax=FMAX, 
            window=WINDOW_TYPE, 
            win_length=WIN_SIZE
        )

        mel_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=[7, 7])
        ax = plt.gca()
        ax.set_axis_off()

        librosa.display.specshow(mel_db, y_axis='mel', fmin=FMIN, fmax=FMAX, x_axis='time')

        plt.savefig(mel_save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        gc.collect()

    def convert_to_mel_spectrogram(self, audio_path, mel_save_path):
        """Convert an audio file to a mel spectrogram and save it."""
        signal, sr = librosa.load(audio_path, sr=None)
        self.saveMel(signal, sr, mel_save_path)

    def save_all_mel_spectrograms(self):
        """Convert all audio files to mel spectrograms."""
        print("Converting audio files to mel spectrograms...")
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            mel_filename = os.path.join(self.mel_spectrograms_dir, row[self.file_col].replace('.wav', '.png'))
            self.convert_to_mel_spectrogram(file_path, mel_filename)

    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        import requests
        file = Path(self.root) / self.filename
        if file.is_file():
            return
        
        r = requests.get(self.url, stream=True)
        tmp = file.with_suffix('.tmp')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'wb') as f:
            pbar = tqdm(unit=" MB", bar_format=f'{file.name}: {{rate_noinv_fmt}}')
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk) / 1024 / 1024)
                    f.write(chunk)
                    
        tmp.rename(file)
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)

from esc50_dataset import ESC50
root_path = "/Users/sebasmos/Documents/VE_paper/data" 
dataset = ESC50(root=root_path, download=False)
dataset.save_all_mel_spectrograms()