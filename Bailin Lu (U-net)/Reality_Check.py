import numpy as np
from PIL import Image
from pathlib import Path

# Find unique classes in YOUR dataset
mask_dir = Path('./data/masks')
unique_classes = set()

for mask_file in mask_dir.glob('*.png'):
    mask = np.array(Image.open(mask_file))
    unique_classes.update(np.unique(mask).tolist())

print(f"Unique classes in YOUR data: {sorted(unique_classes)}")
print(f"Total: {len(unique_classes)} classes")