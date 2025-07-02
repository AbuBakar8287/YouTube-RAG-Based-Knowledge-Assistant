import streamlit as st
import re
import os
from dotenv import load_dotenv

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Cohere for reranking
import cohere

# Load environment variables
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Extract video ID
def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    return None

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit UI
st.title("🎬 YouTube RAG Assistant with Chain Concept")

video_url = st.text_input("Enter YouTube Video URL (with transcript):")
question = st.text_input("Ask a question based on the video content:")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL or ID.")
        st.stop()

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("Transcript fetched successfully! Vector store being created...")

        # Vector store setup
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.5})

        # LLM setup
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Compression retriever
        compressor = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        # Prompt template
        prompt = PromptTemplate(
            template="""You are a helpful assistant.  
                        Answer ONLY from the provided transcript context. 
                        If the context is insufficient, say "I don't know."

                        Context:
                        {context}

                        Question: {question}
                    """,
            input_variables=["context", "question"]
        )

        # Reranking function
        def rerank_docs(docs, query, top_n=4):
            candidates = [doc.page_content for doc in docs]
            response = co.rerank(query=query, documents=candidates, top_n=top_n)
            return [docs[result.index] for result in response.results]

        if question:
            with st.spinner("Generating answer..."):
                
                # Define Parallel Chain
                parallel_chain = RunnableParallel({
                    "context": compression_retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                })

                # Complete Chain with Prompt and LLM
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                # Invoke the chain
                answer = main_chain.invoke(question)

                st.write("### Answer:")
                st.write(answer)

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f"Error: {e}")
