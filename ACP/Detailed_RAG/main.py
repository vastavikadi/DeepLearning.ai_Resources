# A RAG agent with CrewAI by integrating RagTool from crewai_tools with a CrewAI agent. RagTool provides a way to create and query knowledge bases from various data sources, and allows the agent to access specialized context.

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

# “RAG” typically refers to Retrieval-Augmented Generation, a method that combines retrieval from external documents{vector database} with generative models like GPT to improve factual accuracy and grounding.

# A hybrid model that first retrieves relevant documents from a corpus (using dense retrievers like FAISS, BM25, etc.)
# Then uses these retrieved documents as context to generate an answer using a generative model (like GPT or BART).
# It augments generation with retrieval for up-to-date, factual, and verifiable outputs.

# How it typically works:
# 1️⃣ Query
# 2️⃣ Retriever fetches top-k relevant documents →
# 3️⃣ Generator takes the original query + retrieved context →
# 4️⃣ Generates the answer.

# Why use RAG?
# ✅ Better factual grounding
# ✅ Can use smaller generative models while leveraging large corpora
# ✅ Reduces hallucinations compared to pure generation

# Example frameworks/libraries to implement RAG:

# Hugging Face RAG pipeline
# LangChain for orchestration with vector stores + LLMs
# LlamaIndex for flexible document retrieval + prompt engineering
# Haystack for production pipelines with retriever-generator architecture.

llm = LLM(model="openai/gpt-4",max_tokens=1024)
# we will now define the large language model that we will use for the CrewAI agent. max_tokens: maximum number of tokens the model can generate in a single response.
# Note: If you will define this model locally, it requires that you define the API key in a .env file as follows:

# Required
# OPENAI_API_KEY=sk-...
# Optional
# OPENAI_API_BASE=<custom-base-url>
# OPENAI_ORGANIZATION=<your-org-id>


# For the RAG tool, we can define the model provider and the embedding model in a configuration Python dictionary. We can also define the details of the vector database. If you don't specify the vector database, the RagTool will use Chroma (ChromaDB) as the default vector database in local/in-memory mode.

#NOTE: An embedding model is a machine learning model that converts data (like text, images, or audio) into numerical vectors (lists of numbers) in a high-dimensional space. For text, embedding models (such as OpenAI's "text-embedding-ada-002") map words, sentences, or documents to vectors so that similar meanings are close together in this space. This makes it easier for algorithms to compare, search, or cluster similar content.

# A vector database (vector DB) is a specialized database designed to store and search these high-dimensional vectors efficiently. When you want to find documents similar to a query, the vector DB quickly retrieves the most relevant vectors (and their associated data) using similarity search (like cosine similarity). In Retrieval-Augmented Generation (RAG), the embedding model turns your query and documents into vectors, and the vector DB helps retrieve the most relevant documents to provide context for the language model.
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
        },
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
        },
    },
}

# We can then pass the config to the RagTool, and then specify the data source{like a pdf or data context we want to provide to the llm} for which the knowledge base will be constructed. When embedding your data, the RagTool chunks your document into chunks and create an embedding vector for each chunk. You can specify the chunk size (chunk_size: number of characters) and how many characters overlap between consecutive chunks (chunk_overlap){exmaple to understand this: if you split a document into chunks of 1200 characters with a chunk_overlap of 200, each new chunk will start 200 characters before the end of the previous chunk. This ensures that important information at the boundaries of chunks is not lost and helps maintain context continuity across chunks, which improves retrieval and answer quality in RAG systems.}. You can also use the default behavior.

rag_tool = RagTool(config=config, chunk_size=1200, chunk_overlap=200)
rag_tool.add("../AiForEveryone/W1.pdf", data_type="pdf_file")

# Now that we have the rag_tool defined, we define the CrewAI agent that can assist with pdf {or any context} coverage queries.
AI_teacher = Agent(
    role="AI Teacher",
    goal="Teach students about AI concepts",
    backstory="You are an AI teacher who explains complex AI concepts in simple terms.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[rag_tool],
    max_retry_limit=5
)

# Let's now test the insurance agent. For that, we need to define the agent task and pass to it the query and the agent.
task1=Task(
    description="Explain the concept of RAG (Retrieval-Augmented Generation) in AI.",
    expected_output="A clear and concise explanation to the users' questions.",
    agent=AI_teacher
)

# To run the agent, we need to pass the agent and the task to a Crew object that you can run using the kickoff method.
crew = Crew(agents=[AI_teacher], tasks=[task1], verbose=True)
task_output = crew.kickoff()
print(task_output)