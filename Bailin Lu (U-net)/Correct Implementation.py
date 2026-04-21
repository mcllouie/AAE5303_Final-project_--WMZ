# 1. Define valid classes (from dataset analysis)
VALID_CLASS_IDS = [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24, 25]

# 2. Create ID mapping
id_to_idx = {class_id: idx for idx, class_id in enumerate(VALID_CLASS_IDS)}

# 3. Convert dataset labels
def convert_label(original_mask):
    """Convert original class IDs to continuous indices 0-13"""
    new_mask = np.zeros_like(original_mask)
    for class_id, idx in id_to_idx.items():
        new_mask[original_mask == class_id] = idx
    return new_mask

# 4. Create model with correct number
n_classes = len(VALID_CLASS_IDS)  # = 14
model = UNet(n_channels=3, n_classes=n_classes)