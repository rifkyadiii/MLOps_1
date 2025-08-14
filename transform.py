import tensorflow as tf
import tensorflow_transform as tft

# Definisikan fitur-fitur yang akan di-transformasi
NUMERIC_FEATURES = [
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
    'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram',
    'sc_h', 'sc_w', 'talk_time'
]

BINARY_FEATURES = [
    'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'
]

# Definisikan fitur target (label)
LABEL_KEY = 'price_range'

def preprocessing_fn(inputs):
    """
    Fungsi untuk preprocessing data.
    Akan dijalankan oleh komponen TFX Transform.
    """
    outputs = {}

    # Normalisasi fitur numerik dengan Z-score
    for key in NUMERIC_FEATURES:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Fitur biner tidak perlu diubah
    for key in BINARY_FEATURES:
        outputs[key] = inputs[key]

    # Label juga tidak diubah
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs