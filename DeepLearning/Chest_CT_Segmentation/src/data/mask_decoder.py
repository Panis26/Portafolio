import numpy as np

BACKGROUND = 0
LUNG = 1
HEART = 2
TRACHEA = 3

def decode_mask(mask, thr=200, low=80, return_debug=False):
    """
    Decodifica una máscara (H, W, 3) en una máscara de etiquetas (H, W) basada en umbrales de color.

    Este dataser está en JPG, por lo que puede haber artefactos de compresión que afectan la precisión del color.
    Por eso se utilizan umbrales para encontrar el canal dominante:

        - TRACHEA (rojo):   R >= thr y G <= low y B <= low
        - HEART (verde):    G >= thr y R <= low y B <= low
        - LUNGS (azul):     B >= thr y R <= low y G <= low
        - BACKGROUND:       Todos los demás píxeles
    
    Parámetros:
    - mask: np.ndarray
        Máscara de entrada de forma (H, W, 3) con valores de píxeles en el rango [0, 255].
    - thr: int
        Umbral superior para identificar el canal dominante.
    - low: int
        Umbral inferior para identificar los canales no dominantes.
    - return_debug: bool
        Si es True, devuelve un conteo de pixeles para cada clase además de la máscara decodificada.
    
    Retorna:
    - decoded_mask: np.ndarray
        Máscara decodificada de forma (H, W) con etiquetas de clase.
    """
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f"\"mask\" debe tener forma (H, W, 3), pero tiene forma {mask.shape}")

    decoded_mask = np.full(mask.shape[:2], BACKGROUND, dtype=np.uint8)

    r = mask[:, :, 0]
    g = mask[:, :, 1]
    b = mask[:, :, 2]

    trachea_mask = (r >= thr) & (g <= low) & (b <= low)
    heart_mask = (g >= thr) & (r <= low) & (b <= low)
    lung_mask = (b >= thr) & (r <= low) & (g <= low)

    # Si hay solapamientos, se prioriza TRACHEA > HEART > LUNG
    decoded_mask[lung_mask] = LUNG
    decoded_mask[heart_mask] = HEART
    decoded_mask[trachea_mask] = TRACHEA

    if return_debug:
        debug = {
            "thr": thr,
            "low": low,
            "num_background_pixels": np.sum(decoded_mask == BACKGROUND),
            "num_lung_pixels": np.sum(decoded_mask == LUNG),
            "num_heart_pixels": np.sum(decoded_mask == HEART),
            "num_trachea_pixels": np.sum(decoded_mask == TRACHEA),
            "total_pixels": decoded_mask.size
        }
        return decoded_mask, debug
    else:
        return decoded_mask