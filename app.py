import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.llm import get_chatgroq_model
from models.embeddings import get_embedding_model
from utils.vector_store import (
    load_vector_store,
    retrieve_relevant_chunks,
    migrate_collection_embeddings,
    get_collection_embedding_dimension,
    get_embedding_dimension,
)
from utils.web_search import (
    perform_web_search,
    format_search_results,
    should_use_web_search,
)
from utils.logger import default_logger as logger
from config.config import Config


def get_response_mode_prompt(base_prompt: str, response_mode: str = "concise") -> str:
    if response_mode.lower() == "concise":
        mode_instruction = (
            "\n\nIMPORTANT: Provide a CONCISE response. "
            "Keep your answer brief, to the point, and summarized. "
            "Focus on key points only. Aim for 2-3 sentences or a short paragraph maximum. "
            "Avoid unnecessary elaboration or examples unless specifically requested."
        )
    elif response_mode.lower() == "detailed":
        mode_instruction = (
            "\n\nIMPORTANT: Provide a DETAILED response. "
            "Give an expanded, in-depth answer with comprehensive information. "
            "Include relevant examples, explanations, context, and supporting details. "
            "Break down complex topics into clear sections. "
            "Be thorough and cover all important aspects of the question."
        )
    else:
        mode_instruction = ""

    return base_prompt + mode_instruction


def get_chat_response(
    chat_model,
    messages,
    system_prompt,
    vector_store=None,
    use_rag=True,
    use_web_search=False,
    web_search_provider=None,
    web_search_api_keys=None,
    response_mode="concise",
):
    try:
        logger.debug(
            f"Getting chat response (RAG={use_rag}, WebSearch={use_web_search}, Mode={response_mode})"
        )
        # Adjust system prompt based on response mode
        adjusted_prompt = get_response_mode_prompt(system_prompt, response_mode)

        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=adjusted_prompt)]

        # Get the last user message for context retrieval
        user_query = (
            messages[-1]["content"]
            if messages and messages[-1]["role"] == "user"
            else ""
        )

        context_parts = []

        # If RAG is enabled and vector store exists, retrieve relevant context
        if use_rag and vector_store is not None and user_query:
            try:
                relevant_chunks = retrieve_relevant_chunks(
                    vector_store, user_query, k=4
                )
                if relevant_chunks:
                    doc_context = "\n\nRelevant context from documents:\n"
                    for i, chunk in enumerate(relevant_chunks, 1):
                        doc_context += f"\n[{i}] {chunk.page_content}\n"
                    doc_context += "\n---\n"
                    context_parts.append(doc_context)
                    logger.debug(
                        f"Retrieved {len(relevant_chunks)} relevant chunks for RAG"
                    )
            except RuntimeError as e:
                error_msg = str(e)
                if "dimension mismatch" in error_msg.lower():
                    warning_msg = (
                        "**RAG Error: Embedding Dimension Mismatch**\n\n"
                        "The knowledge base was created with a different embedding model. "
                        "Please ask the application administrator to recreate the knowledge base "
                        "with a compatible embedding model. RAG is temporarily disabled."
                    )
                    st.warning(warning_msg)
                    logger.error(f"RAG dimension mismatch: {error_msg}", exc_info=True)
                else:
                    logger.warning(
                        f"RAG retrieval warning: {error_msg}", exc_info=True
                    )
                    st.warning(f"RAG retrieval warning: {error_msg}")
            except Exception as e:
                logger.warning(
                    f"RAG retrieval warning: {str(e)}", exc_info=True
                )
                st.warning(f"RAG retrieval warning: {str(e)}")

        # Perform web search if enabled and needed
        if use_web_search and user_query:
            try:
                should_search = should_use_web_search(
                    user_query,
                    vector_store=vector_store,
                    rag_enabled=use_rag,
                )

                if should_search:
                    with st.spinner("Searching the web..."):
                        search_results = perform_web_search(
                            query=user_query,
                            provider=web_search_provider,
                            max_results=Config.MAX_SEARCH_RESULTS,
                            api_keys=web_search_api_keys,
                        )
                        if search_results:
                            web_search_context = format_search_results(search_results)
                            context_parts.append(web_search_context)
            except Exception as e:
                st.warning(f"Web search warning: {str(e)}")

        # Combine all context
        if context_parts:
            combined_context = "\n".join(context_parts)
            formatted_messages[0] = SystemMessage(
                content=adjusted_prompt
                + "\n\n"
                + combined_context
                + "\nUse the above context to answer the user's question. "
                + "Prioritize information from documents if available, then web search results, "
                + "and finally your general knowledge if the context doesn't contain relevant information."
            )
        else:
            formatted_messages[0] = SystemMessage(content=adjusted_prompt)

        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        # Get response from model
        try:
            response = chat_model.invoke(formatted_messages)
            logger.info("Successfully received response from model")
            return response.content
        except Exception as e:
            logger.error(f"Error invoking chat model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to get response from model: {str(e)}") from e

    except RuntimeError as e:
        logger.error(f"Runtime error in get_chat_response: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in get_chat_response: {str(e)}", exc_info=True)
        return f"Error getting response: {str(e)}"


def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")

    st.markdown(
        """
    ## Installation

    First, install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    ## Free API Key Setup

    This chatbot uses only **FREE** services. Get your API keys from:

    ### 1. Groq (LLM - Free)
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key (free tier available)
    - Set `GROQ_API_KEY` environment variable (e.g., in your `.env` file)

    ### 2. Jina (Embeddings - Free)
    - Visit [Jina AI Embeddings](https://jina.ai/embeddings/)
    - Sign up and get your free API key
    - Set `JINA_API_KEY` environment variable (e.g., in your `.env` file)

    ### 3. Tavily (Web Search - Free)
    - Visit [Tavily](https://tavily.com)
    - Sign up for free tier
    - Get your API key
    - Set `TAVILY_API_KEY` environment variable (e.g., in your `.env` file)

    ## Available Models

    ### Groq Models (Free)
    Check Groq's model docs for available models.
    Popular free models include:
    - `llama-3.1-70b-versatile` - Large, powerful model (default)
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability

    ### Jina Embeddings (Free)
    - `jina-embeddings-v2-base-en` - High-quality embeddings (default)

    ## How to Use

    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!

    ## Tips

    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh

    ## Troubleshooting

    - **API Key Issues**: Make sure your API key is valid
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status

    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """
    )


def initialize_rag_system():
    if "embedding_model" not in st.session_state:
        try:
            embedding_api_key = st.session_state.get("jina_api_key", None)
            st.session_state.embedding_model = get_embedding_model(
                "jina", embedding_api_key
            )
        except Exception as e:
            st.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(
                f"Failed to initialize embedding model: {str(e)}", exc_info=True
            )
            return None

    # Load existing vector store (no creation from UI)
    if "vector_store" not in st.session_state:
        try:
            persist_dir = "./chroma_db"
            vector_store = load_vector_store(
                st.session_state.embedding_model,
                persist_directory=persist_dir,
            )
            st.session_state.vector_store = vector_store
        except ValueError as e:
            error_msg = str(e)
            if "dimension mismatch" in error_msg.lower():
                # Get dimensions for display
                try:
                    persist_dir = "./chroma_db"
                    collection_dim = get_collection_embedding_dimension(
                        persist_directory=persist_dir
                    )
                    model_dim = get_embedding_dimension(
                        st.session_state.embedding_model
                    )

                    st.warning(
                        f"**Embedding Dimension Mismatch Detected**\n\n"
                        f"The existing knowledge base uses {collection_dim}-dimensional embeddings, "
                        f"but your current Jina AI model produces {model_dim}-dimensional embeddings.\n\n"
                        "You can automatically migrate the collection to use the new embedding dimensions."
                    )

                    if st.button(
                        "Migrate Knowledge Base to New Embedding Dimension",
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "Migrating knowledge base... This may take a few minutes."
                        ):
                            try:
                                migrated_store = migrate_collection_embeddings(
                                    st.session_state.embedding_model,
                                    persist_directory="./chroma_db",
                                )
                                st.session_state.vector_store = migrated_store
                                st.success(
                                    f"Successfully migrated knowledge base to {model_dim}-dimensional embeddings! "
                                    "The RAG system is now ready to use."
                                )
                                st.rerun()
                            except Exception as mig_error:
                                st.error(f"Migration failed: {str}(mig_error)")
                                logger.error(
                                    f"Migration error: {str(mig_error)}",
                                    exc_info=True,
                                )
                except Exception:
                    st.warning(
                        "**Embedding Dimension Mismatch Detected**\n\n"
                        "The existing knowledge base was created with a different embedding model. "
                        "Please contact the application administrator to reset or rebuild "
                        "the knowledge base (e.g., by recreating the `./chroma_db` directory) "
                        "with the current embedding model."
                    )
                logger.error(f"Dimension mismatch: {error_msg}")
            else:
                st.error(f"Error loading vector store: {error_msg}")
            st.session_state.vector_store = None
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            st.session_state.vector_store = None

    return st.session_state.get("vector_store")


def chat_page():
    st.title("Metro Hospitals HR Policy Assistant")

    # Ensure session state is initialized from environment variables
    st.session_state.setdefault("groq_api_key", Config.get_groq_api_key())
    st.session_state.setdefault("jina_api_key", Config.get_jina_api_key())
    st.session_state.setdefault("tavily_api_key", Config.get_tavily_api_key())

    # Initialize RAG system (load existing vector store)
    vector_store = initialize_rag_system()

    # Sidebar configuration
    with st.sidebar:
        st.header("Response Mode")
        response_mode = st.radio(
            "Select Response Style",
            ["Concise", "Detailed"],
            index=0,
            help="Concise: Short, summarized replies | Detailed: Expanded, in-depth responses",
        )
        st.session_state.response_mode = response_mode.lower()

        if response_mode == "Concise":
            st.caption("Short, summarized replies (2-3 sentences)")
        else:
            st.caption("Expanded, in-depth responses with examples and details")

        st.divider()

        st.header("RAG Configuration")

        # RAG toggle
        use_rag = st.checkbox(
            "Enable RAG",
            value=True,
            help="Use document retrieval for responses (existing knowledge base only)",
        )
        st.session_state.use_rag = use_rag

        # Vector store status (read-only)
        if vector_store is not None:
            st.success("Knowledge base loaded")
        else:
            st.info(
                "No knowledge base loaded. RAG responses will rely on general knowledge and web search."
            )

        st.divider()

        # Web Search Configuration
        st.header("Web Search Configuration")

        use_web_search = st.checkbox(
            "Enable Web Search",
            value=False,
            help="Perform real-time web searches when knowledge base lacks information",
        )
        st.session_state.use_web_search = use_web_search

        if use_web_search:
            tavily_key = st.session_state.get("tavily_api_key")
            st.session_state.web_search_api_keys = {"tavily": tavily_key} if tavily_key else None
            st.session_state.web_search_provider = "tavily"
        else:
            st.session_state.web_search_api_keys = None
            st.session_state.web_search_provider = None

    # Groq API Key Configuration (sidebar)
    st.sidebar.divider()

    # Default system prompt (will be adjusted based on response mode)
    system_prompt = "You are a helpful AI assistant. Answer questions accurately and helpfully."

    # Initialize chat model
    try:
        groq_key = st.session_state.get("groq_api_key", None)
        chat_model = get_chatgroq_model(api_key=groq_key)
        logger.info("Chat model initialized successfully")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Failed to initialize chat model: {str(e)}")
        st.error(
            f"Chat model initialization failed: {str(e)}. "
            "Please set `GROQ_API_KEY` in your `.env` file or environment."
        )
        chat_model = None

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if chat_model is None:
            st.error("Chat model is not initialized. Please configure your API keys.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"User message received: {prompt[:50]}...")

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                try:
                    use_rag_flag = st.session_state.get("use_rag", True)
                    use_web_search_flag = st.session_state.get(
                        "use_web_search", False
                    )
                    web_search_provider = st.session_state.get(
                        "web_search_provider", None
                    )
                    web_search_api_keys = st.session_state.get(
                        "web_search_api_keys", None
                    )
                    response_mode = st.session_state.get("response_mode", "concise")

                    response = get_chat_response(
                        chat_model,
                        st.session_state.messages,
                        system_prompt,
                        vector_store=vector_store if use_rag_flag else None,
                        use_rag=use_rag_flag,
                        use_web_search=use_web_search_flag,
                        web_search_provider=web_search_provider,
                        web_search_api_keys=web_search_api_keys,
                        response_mode=response_mode,
                    )

                    if response and response.strip():
                        st.markdown(response)
                        logger.info("Response generated successfully")

                        # Add bot response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        error_msg = (
                            "Received empty response from the model. Please try again."
                        )
                        logger.warning(error_msg)
                        st.warning(error_msg)
                except Exception as e:
                    error_msg = f"Failed to generate response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    # Don't add error to chat history


def main():
    st.set_page_config(
        page_title="AI RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation and global sidebar items
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0,
        )

        if page == "Chat":
            st.divider()
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()