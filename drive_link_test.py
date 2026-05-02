import re
import requests

def extract_folder_id(drive_link: str):
    """Extract folder ID from a Google Drive share link."""
    match = re.search(r"[-\w]{25,}", drive_link)
    return match.group(0) if match else None

def list_public_images(folder_id: str):
    """List publicly shared image files inside a Drive folder (no auth)."""
    import os; api_key = os.getenv("GOOGLE_API_KEY", "")
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents+and+(mimeType contains 'image/')&key={api_key}&fields=files(id,name,mimeType)"
    res = requests.get(url)
    if res.status_code != 200:
        print("Error:", res.text)
        return []
    data = res.json()
    return data.get("files", [])

if __name__ == "__main__":
    drive_link = input("Enter your shared folder link: ")
    folder_id = extract_folder_id(drive_link)
    if not folder_id:
        print("Could not extract folder ID. Please check the link.")
    else:
        print("Folder ID:", folder_id)
        images = list_public_images(folder_id)
        print("\nFound image files:")
        for img in images:
            print(f"- {img['name']} ({img['id']})")
