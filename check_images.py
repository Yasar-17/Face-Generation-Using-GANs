from pathlib import Path
from PIL import Image

folder = Path(r"C:\Users\HP\OneDrive\Pictures\Faces")

bad = []

for img_path in folder.iterdir():
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"Broken: {img_path.name}")
            bad.append(img_path.name)

print("\nDone.")
print(f"Broken images: {len(bad)}")