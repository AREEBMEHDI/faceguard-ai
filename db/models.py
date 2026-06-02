from sqlalchemy import create_engine, Column, String, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/ai_face_alert")

engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    label = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Detection(Base):
    __tablename__ = "detections"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_name = Column(String)
    camera = Column(String)
    confidence = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_level = Column(String)
    action = Column(String)
    reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
