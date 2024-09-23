import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from OllamaWrapper import OllamaLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


def get_query_engine(filename):
    nest_asyncio.apply()

    documents = SimpleDirectoryReader(input_files=[filename]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    Settings.llm = OllamaLLM(modelname="llama3.1")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    vector_query_engine = vector_index.as_query_engine()
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )

    return query_engine

if __name__ == "__main__":
    query_engine = get_query_engine("metagpt.pdf")
    query = "What is the MetaGPT paper about?"
    response = query_engine.query(query)
    print(response)
