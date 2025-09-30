import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from a .env file
# This allows you to keep your database password and other secrets out of the code.
load_dotenv()

# Get the database URL from environment variables, with a default for local development.
# Example .env entry: DATABASE_URL="postgresql://your_user:your_password@localhost/minedb"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/minedb")

# The engine is the central point of communication with the database.
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory for creating new database sessions (i.e., conversations with the DB).
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is a class that all our database models will inherit from.
Base = declarative_base()

# This is a FastAPI "dependency".
# It creates a new database session for each incoming API request and ensures it's
# properly closed afterward, even if an error occurs.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

