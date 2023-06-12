import pdf2image
import os
from PIL import Image

folder_to_convert = "./paper_ready_plots/power_narrowing/"

def pdf_to_eps(pdf_path, eps_path):
    # Convert PDF to images
    images = pdf2image.convert_from_path(pdf_path)

    # Save images in EPS format
    for i, image in enumerate(images):
        eps_file = f"{eps_path}_{i+1}.eps"
        image = image.convert("RGB")

        image.save(eps_file, "EPS", quality=10, optimize=True)

for f in os.listdir(folder_to_convert):
    pdf_path = os.path.join(folder_to_convert, f)
    eps_path = pdf_path[:-4]
    pdf_to_eps(pdf_path, eps_path)
