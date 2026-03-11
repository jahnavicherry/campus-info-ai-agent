import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.vector_store import create_vector_db

if __name__ == "__main__":
    create_vector_db()