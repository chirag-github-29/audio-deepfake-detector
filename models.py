from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True)
    username   = Column(String(50), unique=True)
    email      = Column(String(100), unique=True)
    password   = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

class Detection(Base):
    __tablename__ = "detections"

    id         = Column(Integer, primary_key=True)
    filename   = Column(String(200))
    prediction = Column(String(10))
    confidence = Column(Float)
    label      = Column(Integer)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    owner_id   = Column(Integer)