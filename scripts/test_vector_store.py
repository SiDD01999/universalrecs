"""
Test script to verify vector store setup without running full embedding generation.
This is useful for checking if all dependencies are properly installed.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required libraries can be imported."""
    print("=" * 60)
    print("Testing Vector Store Dependencies")
    print("=" * 60)

    try:
        print("\n[1/6] Testing pandas import...")
        import pandas as pd
        print("  [OK] pandas imported successfully")

        print("\n[2/6] Testing numpy import...")
        import numpy as np
        print("  [OK] numpy imported successfully")

        print("\n[3/6] Testing chromadb import...")
        import chromadb
        print("  [OK] chromadb imported successfully")

        print("\n[4/6] Testing torch import...")
        import torch
        print(f"  [OK] torch imported successfully (version: {torch.__version__})")

        print("\n[5/6] Testing sentence-transformers import...")
        from sentence_transformers import SentenceTransformer
        print("  [OK] sentence-transformers imported successfully")

        print("\n[6/6] Testing vector_store module import...")
        from src.vector_store import MovieVectorStore
        print("  [OK] vector_store module imported successfully")

        print("\n" + "=" * 60)
        print("All dependencies installed correctly!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python scripts/generate_embeddings.py --reset")
        return True

    except ImportError as e:
        print(f"\n[ERROR] Import Error: {e}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False
    except OSError as e:
        print(f"\n[ERROR] OS Error: {e}")
        if "DLL" in str(e) or "c10.dll" in str(e):
            print("\n[WARNING] Microsoft Visual C++ Redistributable is required!")
            print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("   Install it and run this test again.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
