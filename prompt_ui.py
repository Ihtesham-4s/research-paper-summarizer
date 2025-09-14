from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

# Model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model = ChatHuggingFace(llm=llm)

# UI
st.header("📄 Research Paper Summarizer")

# Paper selection
paper_choice = st.selectbox(
    "Select Research Paper", 
    [
        "Attention is All You Need", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis", 
        "Other (type your own)"
    ]
)

if paper_choice == "Other (type your own)":
    paper_input = st.text_input("Enter research paper title or link")
else:
    paper_input = paper_choice

# Writing style
style_input = st.selectbox(
    "Select Writing Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Length
length_input = st.selectbox(
    "Select Summary Length", 
    ["Short (1-2 Paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load template
template = load_prompt("research_summary_template.json")

# Fill placeholders
prompt = template.invoke(
    {"paper_input": paper_input, "style_input": style_input, "length_input": length_input}
)

# Generate summary
if st.button("Summarize"):
    if not paper_input:
        st.warning("⚠️ Please enter a paper title or link.")
    else:
        response = model.invoke(prompt)
        st.subheader("📝 Summary")
        st.write(response.content)

        # Simple recommendations
        st.subheader("🔍 You may also like:")
        if "Attention is All You Need" in paper_input:
            st.markdown("- BERT: Pre-training of Deep Bidirectional Transformers")
            st.markdown("- GPT-3: Language Models are Few-Shot Learners")
        elif "BERT" in paper_input:
            st.markdown("- RoBERTa: A Robustly Optimized BERT Pretraining Approach")
            st.markdown("- DistilBERT: Smaller, Faster, Cheaper BERT")
        elif "GPT-3" in paper_input:
            st.markdown("- GPT-4 Technical Report")
            st.markdown("- LLaMA: Open and Efficient Foundation Language Models")
        else:
            st.markdown("- Explore arXiv for related works")
