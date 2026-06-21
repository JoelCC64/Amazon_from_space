import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

def visualizar_imagen(img_path, title=""):
    """
    To load and display an image using matplotlib with a descriptive title.
    """
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}")
        return
    img = Image.open(img_path)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def ejecutar_inferencia_test(modelo, test_img_path, archivos_test, transformaciones, idx_to_tag, device, start_idx=9, end_idx=12, umbral=0.5, show_images=True):
    """
    To run inference on a subset of test dataset images and
    display tag predictions alongside the images (optional).
    """
    modelo.eval()
    print(f'\n--- Starting inference in range [{start_idx}:{end_idx}] ---')
    
    with torch.no_grad():
        for i in range(start_idx, min(end_idx, len(archivos_test))):
            nombre_archivo = archivos_test[i]
            ruta_imagen = os.path.join(test_img_path, nombre_archivo)
            
            if not os.path.exists(ruta_imagen):
                print(f"Test image not found at: {ruta_imagen}")
                continue

            img_pil = Image.open(ruta_imagen).convert('RGB')
            tensor_img = transformaciones(img_pil)
            tensor_img = tensor_img.unsqueeze(0).to(device)

            logits = modelo(tensor_img)
            probabilidades = torch.sigmoid(logits).squeeze(0)

            indices_activados = (probabilidades > umbral).nonzero(as_tuple=True)[0].cpu().tolist()
            etiquetas_predichas = [idx_to_tag[idx] for idx in indices_activados]
            
            print(f"\nImage: {nombre_archivo}")
            print(f"Predicted Labels: {etiquetas_predichas}")
            
            if show_images:
                visualizar_imagen(ruta_imagen, title=f"Predictions: {etiquetas_predichas}")
