from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello! Server is running ✅"}

@app.post("/upload-selfie/")
async def upload_selfie(file: UploadFile = File(...)):
    return {"filename": file.filename}
