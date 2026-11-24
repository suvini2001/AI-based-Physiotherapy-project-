import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Read the database URL from the environment.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./physio.db")
# Create the connection engine.
engine = create_engine(DATABASE_URL)

# Factory for creating database sessions.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for SQLAlchemy database models.
Base = declarative_base()


def get_db():
    """Provide a database session for each API request."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()