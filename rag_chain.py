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
        return "I couldn't find any information about that query. Please try different wording or check spelling."

    # Step 2: Combine the relevant documents' text
    full_context = ""
    for doc in relevant_docs:
        full_context += f"{doc.page_content}\n\n"

    # Step 3: Limit the context length to avoid token overflow
    # Tokenize full context separately
    context_tokens = tokenizer(full_context, return_tensors="pt", truncation=False)["input_ids"][0]

    max_context_tokens = 1024  # max tokens to keep in context (adjust as needed)
    if len(context_tokens) > max_context_tokens:
        # Truncate context tokens to last max_context_tokens tokens (keep recent context)
        context_tokens = context_tokens[-max_context_tokens:]
        # Decode truncated tokens back to string for prompt
        truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    else:
        truncated_context = full_context

    # Step 4: Build the prompt with truncated context
    prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{truncated_context}

Question: {user_question}

Answer:"""

    # Step 5: Tokenize prompt and move to model device, with truncation enabled just in case
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Step 6: Generate response
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Step 7: Decode full output text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Step 8: Extract just the answer by removing the prompt part
    answer_start = output_text.find("Answer:")
    if answer_start != -1:
        answer_text = output_text[answer_start + len("Answer:"):].strip()
    else:
        answer_text = output_text

    return answer_text



