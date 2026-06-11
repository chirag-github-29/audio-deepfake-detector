from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import engine, get_db, Base
from auth import (
    hash_password, verify_password,
    create_access_token, get_current_user
)
import models
import schemas
import librosa
import numpy as np
import joblib
import os
import shutil
import warnings
warnings.filterwarnings("ignore", ".*error reading bcrypt version.*")
warnings.filterwarnings("ignore", ".*trapped error reading bcrypt version.*")

# ── Create Tables ────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Load Model ───────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ml", "model.pkl")
model      = joblib.load(model_path)

# ── App ──────────────────────────────────────────
app = FastAPI(title="Audio Deepfake Detector API")

# ── Feature Extraction ───────────────────────────
def extract_features(file_path: str):
    audio, sr = librosa.load(
        file_path, sr=16000, duration=3.0
    )

    max_len = 16000 * 3
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))

    mfcc     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y=audio)
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr)
    rms      = librosa.feature.rms(y=audio)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(centroid)],
        [np.mean(rolloff)],
        [np.mean(zcr)],
        np.mean(chroma, axis=1),
        [np.mean(rms)]
    ])

    return features

# ── Auth Routes ──────────────────────────────────
@app.post("/auth/register", response_model=schemas.UserResponse)
def register(
    user : schemas.UserCreate,
    db   : Session = Depends(get_db)
):
    if db.query(models.User).filter(
        models.User.username == user.username
    ).first():
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    if db.query(models.User).filter(
        models.User.email == user.email
    ).first():
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    new_user = models.User(
        username = user.username,
        email    = user.email,
        password = hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/login", response_model=schemas.Token)
def login(
    credentials : OAuth2PasswordRequestForm = Depends(),
    db          : Session = Depends(get_db)
):
    user = db.query(models.User).filter(
        models.User.username == credentials.username
    ).first()

    if not user or not verify_password(
        credentials.password, user.password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type"  : "bearer"
    }

# ── Detect Route ─────────────────────────────────
@app.post("/api/detect/")
def detect(
    audio_file   : UploadFile    = File(...),
    db           : Session       = Depends(get_db),
    current_user : models.User   = Depends(get_current_user)
):
    # Save uploaded file temporarily
    temp_dir  = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, audio_file.filename)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    try:
        # Extract features
        features = extract_features(temp_path)

        # Get probability
        proba      = model.predict_proba([features])[0][1]
        label      = 1 if proba >= 0.5 else 0
        prediction = "REAL" if label == 1 else "FAKE"
        confidence = round(float(proba), 4)

        # Save to database
        detection = models.Detection(
            filename   = audio_file.filename,
            prediction = prediction,
            confidence = confidence,
            label      = label,
            owner_id   = current_user.id
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)

        return {
            "prediction" : prediction,
            "confidence" : confidence,
            "label"      : label,
            "filename"   : audio_file.filename,
            "timestamp"  : detection.timestamp
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ── History Route ────────────────────────────────
@app.get("/api/history/")
def history(
    db           : Session     = Depends(get_db),
    current_user : models.User = Depends(get_current_user)
):
    detections = db.query(models.Detection).filter(
        models.Detection.owner_id == current_user.id
    ).order_by(
        models.Detection.timestamp.desc()
    ).limit(20).all()

    return detections

# ── Health Check ─────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Audio Deepfake Detector API",
        "status" : "running",
        "docs"   : "/docs"
    }