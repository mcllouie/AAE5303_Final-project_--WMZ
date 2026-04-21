import os

img_dir = 'data/imgs/'
mask_dir = 'data/masks/'

# Get sorted lists of files to ensure they line up chronologically
imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])

# Safety check
if len(imgs) != len(masks):
    print(f"CRITICAL ERROR: You have {len(imgs)} images but {len(masks)} masks!")
    print("The numbers must match exactly. Please check your folders.")
else:
    print(f"Found {len(imgs)} images and masks. Formatting names...")
    
    for i, (img_name, mask_name) in enumerate(zip(imgs, masks)):
        # Create a clean, padded number (e.g., "000001", "000002")
        new_base = f"{i:06d}"
        
        # Keep original file extensions
        img_ext = os.path.splitext(img_name)[1]
        mask_ext = os.path.splitext(mask_name)[1]
        
        # Set the new names (adding _mask so the Carvana loader finds it instantly)
        new_img_name = f"{new_base}{img_ext}"
        new_mask_name = f"{new_base}_mask{mask_ext}"
        
        # Rename the files
        os.rename(os.path.join(img_dir, img_name), os.path.join(img_dir, new_img_name))
        os.rename(os.path.join(mask_dir, mask_name), os.path.join(mask_dir, new_mask_name))

    print("Successfully formatted all files! You are ready to train.")