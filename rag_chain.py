import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import re

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
    """Lazy load model and tokenizer - using a much smaller model"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading smaller model...")
        try:
            # Use a much smaller model that's more suitable for Streamlit Cloud
            model_name = "microsoft/DialoGPT-small"  # Only 117M parameters
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            print("✅ Small model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    return _tokenizer, _model

def extract_drug_food_info(context):
    """Extract drug and food interaction information from context"""
    interactions = []
    
    # Look for interaction patterns in the context
    lines = context.split('\n')
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['interaction', 'avoid', 'limit', 'caution', 'warning']):
            interactions.append(line)
    
    return interactions

def create_template_response(user_question, relevant_docs):
    """Create a template-based response when model fails to load"""
    # Extract drugs and foods from the question
    drugs_match = re.search(r'medications?:\s*([^.]+)', user_question, re.IGNORECASE)
    foods_match = re.search(r'foods?:\s*([^.]+)', user_question, re.IGNORECASE)
    
    drugs = drugs_match.group(1).strip() if drugs_match else "your medications"
    foods = foods_match.group(1).strip() if foods_match else "your foods"
    
    # Extract relevant information from documents
    interactions = []
    for doc in relevant_docs:
        content = doc.page_content.lower()
        if any(keyword in content for keyword in ['interaction', 'avoid', 'limit', 'caution']):
            interactions.append(doc.page_content)
    
    if interactions:
        response = f"Based on the medical information available, here are the key points about interactions between {drugs} and {foods}:\n\n"
        for i, interaction in enumerate(interactions[:3], 1):  # Limit to top 3
            response += f"{i}. {interaction}\n\n"
        response += "⚠️ Always consult your healthcare provider for personalized medical advice."
    else:
        response = f"I found information about {drugs} and {foods}, but no specific interaction warnings were identified in the available medical data. However, it's always best to consult your healthcare provider about potential interactions."
    
    return response

def run_rag_query(user_question: str):
    """Run RAG query with error handling and fallback"""
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

        # Step 3: Try to get model and tokenizer
        tokenizer, model = get_model_and_tokenizer()
        
        # If model loading fails, use template-based response
        if tokenizer is None or model is None:
            print("Model loading failed, using template-based response")
            return create_template_response(user_question, relevant_docs)

        # Step 4: Tokenize context and truncate if too long
        context_tokens = tokenizer(full_context, return_tensors="pt", truncation=False)["input_ids"][0]
        max_context_tokens = 512  # Reduced for smaller model
        if len(context_tokens) > max_context_tokens:
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

        # Step 6: Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Step 7: Generate model output
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,  # Reduced for smaller model
            temperature=0.3,
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
        # Fallback to template-based response
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                relevant_docs = vectorstore.similarity_search(user_question, k=4)
                return create_template_response(user_question, relevant_docs)
        except:
            pass
        
        return f"Sorry, I encountered an error while processing your request. Please try again later."



