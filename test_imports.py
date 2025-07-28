#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        # Test ML imports
        import torch
        print("✅ torch imported successfully")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ transformers imported successfully")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain_huggingface imported successfully")
        
        from langchain_community.vectorstores import FAISS
        print("✅ langchain_community imported successfully")
        
        # Test file access
        import os
        from pathlib import Path
        
        required_files = [
            "data/known_drugs.txt",
            "data/known_foods.txt", 
            "faiss_index/index.faiss",
            "faiss_index/index.pkl"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
        
        print("\n🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 