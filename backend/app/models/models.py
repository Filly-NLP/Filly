from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), default="Untitled")
    content = Column(Text, default="")
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    suggestions = relationship("Suggestion", back_populates="document", cascade="all, delete-orphan")


class Suggestion(Base):
    __tablename__ = "suggestions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    type = Column(String(20), nullable=False)  # 'normalization' or 'grammar'
    original = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=False)
    start_pos = Column(Integer, default=0)
    end_pos = Column(Integer, default=0)
    accepted = Column(Boolean, default=False)
    ignored = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="suggestions")
