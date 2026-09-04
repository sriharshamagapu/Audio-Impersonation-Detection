import os
import sys
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from ml-service
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import VoiceCloneGuardPipeline


app = FastAPI(
    title="Audio Impersonation Detection API",
    version="1.0.0"
)


# Allow Praveen's frontend/backend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("\nLoading Voice Clone Guard pipeline...")
pipeline = VoiceCloneGuardPipeline()
print("Pipeline ready.\n")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Audio Impersonation Detection ML Service"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "AASIST + SSL + Replay + Fusion + Risk"
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file provided"
        )

    # Save uploaded audio temporarily
    suffix = os.path.splitext(file.filename)[1] or ".wav"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            content = await file.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded audio file is empty"
                )

            temp_file.write(content)
            temp_path = temp_file.name

        # Run your existing ML pipeline
        result = pipeline.analyze(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # Delete temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)