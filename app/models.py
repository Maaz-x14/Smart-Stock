# SQLAlchemy models (ShelfLifeReference + NormalizationCache)
# app/models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, func, Date, Boolean, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ShelfLifeReference(Base):
    __tablename__ = "shelf_life_reference"

    id                   = Column(Integer, primary_key=True)
    canonical_name       = Column(String(100), nullable=False)
    category             = Column(String(50),  nullable=False)
    storage_context      = Column(String(20),  nullable=False)
    shelf_life_days_min  = Column(Integer,     nullable=False)
    shelf_life_days_avg  = Column(Integer,     nullable=False)
    shelf_life_days_max  = Column(Integer,     nullable=False)
    notes                = Column(Text)


class NormalizationCache(Base):
    __tablename__ = "normalization_cache"

    id             = Column(Integer, primary_key=True)
    raw_token      = Column(String(200), nullable=False, unique=True)
    canonical_name = Column(String(100), nullable=False)
    source         = Column(String(20),  nullable=False, default="llm")
    created_at     = Column(DateTime,    nullable=False, server_default=func.now())
    hit_count      = Column(Integer,     nullable=False, default=1)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    name = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())