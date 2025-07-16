import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pickle

# Load embedding model
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector store
print("Loading FAISS vector store...")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load TinyLlama model
print("Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print("✅ TinyLlama model loaded.")


def run_rag_query(user_question: str):
    print("\n=== MODEL OUTPUT ===")

    # Step 1: Find relevant documents using vector similarity
    relevant_docs = vectorstore.similarity_search(user_question, k=4)
    if not relevant_docs:
        return "I couldn't find any information about that query. Please try different wording or make sure the medications/foods are spelled correctly."

    context = ""
    for doc in relevant_docs:
        context += f"{doc.page_content}\n\n"


    # Step 3: Create prompt
    prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{context}

Question: {user_question}

Answer:"""

    # Step 4: Tokenize with attention mask and move to model device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Step 5: Generate response
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Step 6: Decode full output text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Step 7: Extract just the answer by removing the prompt part
    answer_start = output_text.find("Answer:")
    if answer_start != -1:
        # +7 to skip the word "Answer:" itself
        answer_text = output_text[answer_start + len("Answer:"):].strip()
    else:
        # fallback: return whole output if "Answer:" not found
        answer_text = output_text

    return answer_text



