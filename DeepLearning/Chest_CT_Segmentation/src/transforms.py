# src/data/transforms.py
import cv2 
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sampple):
        for t in self.transforms:
            sampple = t(sampple)
        return sampple
    
class Resize:
    """
    Redimensiona la imagen y la máscara a un tamaño específico:
    - image: usa interpolación bilineal (bilinear)
    - mask: usa interpolación de vecino más cercano (nearest neighbor)
    """
    def __init__(self, size):
        # size = (H, W)
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        target_h, target_w = self.size

        # Procesar imagen (H, W) o (H, W, 1)
        if img.ndim == 3 and img.shape[2] == 1:
            img2d = img[:, :, 0]
            img_resized = cv2.resize(img2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img_resized = img_resized[:, :, None] 
        elif img.ndim == 2:
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"La imagen debe tener forma (H, W) o (H, W, 1), pero tiene forma {img.shape}")
        
        # Procesar máscara (H, W) con clases 0..3
        if mask.ndim == 2:
            mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"La máscara debe tener forma (H, W), pero tiene forma {mask.shape}")
        
        sample['image'] = img_resized
        sample['mask'] = mask_resized
        return sample
    
    class Normalize:
        """
        Normaliza la imagen a un rango [0, 1].
        Suponiendo que la imagen de entrada tiene valores en el rango [0, 255].
        """
        def __call__(self, sample):
            img = sample['image']
            sample['image'] = img.astype(np.float32) / 255.0
            return sample
    
    class ChannelFirst:
        """
        Convierte la imagen de formato (H, W) o (H, W, 1) a (1, H, W).
        """
        def __call__(self, sample):
            img = sample['image']

            # (H, W) -> (1, H, W)
            if img.ndim == 2:
                img = img[None, :, :] 

            # (H, W, 1) -> (1, H, W)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.transpose(2, 0, 1)

            else:
                raise ValueError(f"La imagen debe tener forma (H, W) o (H, W, 1), pero tiene forma {img.shape}")
            
            sample['image'] = img
            return sample
    
    class maskStats:
        """
        Agrega estadísticas de la máscara al diccionario de muestra para análisis posterior:
        - sample['fg_ratio']: proporción de píxeles diferentes de fondo (clase 0).
        """
        def __init__(self, bgd_value=0):
            self.bgd_value = bgd_value
        
        def __call__(self, sample):
            mask = sample['mask']
            fg = (mask != self.bgd_value)
            sample['fg_ratio'] = float(np.mean(fg))
            return sample

