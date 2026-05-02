from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image
import numpy as np

def load_image_bytes(url):
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Failed to fetch image: {res.status_code}")
    img = Image.open(BytesIO(res.content)).convert("RGB")
    return np.array(img)

if __name__ == "__main__":
    import os
    API_KEY = os.getenv("GOOGLE_API_KEY", "")
    selfie_path = input("Enter path to your selfie (e.g., selfie.jpg): ")
    file_id = input("Enter a Drive file ID to test: ")

    drive_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"
    img_array = load_image_bytes(drive_url)

    result = DeepFace.verify(
        img1_path=selfie_path,
        img2_path=img_array,
        model_name="ArcFace",
        enforce_detection=False
    )

    print("✅ Match!" if result["verified"] else "❌ Not the same person.")
