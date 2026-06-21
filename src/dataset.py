import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from src.config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

def get_unique_tags(df):
    """
    To extract unique tags from the 'tags' column of the DataFrame and
    generate mapping dictionaries between tags and indices.
    """
    lista_de_etiquetas = df['tags'].str.split(' ').explode()
    etiquetas_unicas = sorted(list(lista_de_etiquetas.unique()))
    tag_to_idx = {tag: idx for idx, tag in enumerate(etiquetas_unicas)}
    idx_to_tag = {idx: tag for idx, tag in enumerate(etiquetas_unicas)}
    return etiquetas_unicas, tag_to_idx, idx_to_tag

def codificar_tags_a_tensor(cadena_tags, tag_to_idx):
    """
    To encode a space-separated string of tags into a One-Hot tensor.
    """
    tensor_one_hot = torch.zeros(len(tag_to_idx), dtype=torch.float32)
    etiquetas = cadena_tags.split(' ')
    for tag in etiquetas:
        if tag in tag_to_idx:
            idx = tag_to_idx[tag]
            tensor_one_hot[idx] = 1.0
    return tensor_one_hot

class AmazonDataset(Dataset):
    """
    Custom PyTorch Dataset to load satellite images and encode their multi-label tags.
    """
    def __init__(self, df, img_dir, tag_to_idx, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.tag_to_idx = tag_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _codifier(self, etiquetas):
        encoded_tensor = torch.zeros(len(self.tag_to_idx), dtype=torch.float32)
        for tag in etiquetas.split(' '):
            if tag in self.tag_to_idx:
                encoded_tensor[self.tag_to_idx[tag]] = 1.0
        return encoded_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        image_path = os.path.join(self.img_dir, img_name + '.jpg')
        imagen = Image.open(image_path).convert('RGB')

        image_tags = row['tags']
        tensor_tags = self._codifier(image_tags)

        if self.transform:
            imagen = self.transform(imagen)

        return imagen, tensor_tags

def get_transforms():
    """
    To return the standard image transformation pipeline for ResNet50.
    """
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
