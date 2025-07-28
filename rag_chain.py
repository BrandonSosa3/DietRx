import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading FAISS vector store...")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

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

    # Step 1: Retrieve top 4 relevant documents from FAISS index
    relevant_docs = vectorstore.similarity_search(user_question, k=4)
    if not relevant_docs:
        return "I couldn't find any information about that query. Please try different wording or check spelling."

    # Step 2: Concatenate retrieved document content as context
    full_context = ""
    for doc in relevant_docs:
        full_context += f"{doc.page_content}\n\n"

    # Step 3: Tokenize context and truncate if too long to avoid model input overflow
    context_tokens = tokenizer(full_context, return_tensors="pt", truncation=False)["input_ids"][0]
    max_context_tokens = 1024
    if len(context_tokens) > max_context_tokens:
        # Keep only the last max_context_tokens tokens for recent info
        context_tokens = context_tokens[-max_context_tokens:]
        truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    else:
        truncated_context = full_context

    # Step 4: Build prompt with truncated context and question
    prompt = f"""Using only the relevant medical information below, provide a clear and concise answer to the question. Do NOT repeat the data or the question.

Relevant Info:
{truncated_context}

Question: {user_question}

Answer:"""

    # Step 5: Tokenize prompt and move to model device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Step 6: Generate model output
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Step 7: Decode generated output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Step 8: Extract answer after the "Answer:" keyword
    answer_start = output_text.find("Answer:")
    if answer_start != -1:
        answer_text = output_text[answer_start + len("Answer:"):].strip()
    else:
        answer_text = output_text

    return answer_text



