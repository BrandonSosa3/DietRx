import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Global variables for lazy loading
_embeddings = None
_vectorstore = None
_tokenizer = None
_model = None

def get_embeddings():
    """Lazy load embeddings"""
    global _embeddings
    if _embeddings is None:
        print("Loading embeddings...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_vectorstore():
    """Lazy load vectorstore"""
    global _vectorstore
    if _vectorstore is None:
        print("Loading FAISS vector store...")
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return _vectorstore

def get_model_and_tokenizer():
    """Lazy load model and tokenizer"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading TinyLlama model...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            _model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ TinyLlama model loaded.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    return _tokenizer, _model

def run_rag_query(user_question: str):
    """Run RAG query with error handling"""
    try:
        print("\n=== MODEL OUTPUT ===")

        # Step 1: Retrieve top 4 relevant documents from FAISS index
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "❌ Could not load the knowledge base. Please try again later."
            
        relevant_docs = vectorstore.similarity_search(user_question, k=4)
        if not relevant_docs:
            return "I couldn't find any information about that query. Please try different wording or check spelling."

        # Step 2: Concatenate retrieved document content as context
        full_context = ""
        for doc in relevant_docs:
            full_context += f"{doc.page_content}\n\n"

        # Step 3: Get model and tokenizer
        tokenizer, model = get_model_and_tokenizer()
        if tokenizer is None or model is None:
            return "❌ Could not load the AI model. Please try again later."

        # Step 4: Tokenize context and truncate if too long to avoid model input overflow
        context_tokens = tokenizer(full_context, return_tensors="pt", truncation=False)["input_ids"][0]
        max_context_tokens = 1024
        if len(context_tokens) > max_context_tokens:
            # Keep only the last max_context_tokens tokens for recent info
            context_tokens = context_tokens[-max_context_tokens:]
            truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        else:
            truncated_context = full_context

        # Step 5: Build prompt with truncated context and question
        prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{truncated_context}

Question: {user_question}

Answer:"""

        # Step 6: Tokenize prompt and move to model device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Step 7: Generate model output
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Step 8: Decode generated output
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Step 9: Extract answer after the "Answer:" keyword
        answer_start = output_text.find("Answer:")
        if answer_start != -1:
            answer_text = output_text[answer_start + len("Answer:"):].strip()
        else:
            answer_text = output_text

        return answer_text

    except Exception as e:
        print(f"❌ Error in run_rag_query: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"



