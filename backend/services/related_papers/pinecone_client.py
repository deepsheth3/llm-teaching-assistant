from pinecone import Pinecone, ServerlessSpec
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

# OpenAI embedding dimension (text-embedding-3-small)
EMBEDDING_DIMENSION = 1536


def get_pinecone_index():
    """
    Initialize Pinecone and return the index.

    This function is idempotent:
    - Safe to call multiple times
    - Creates the index only if missing
    """
    settings = get_settings()

    # Initialize Pinecone client
    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_name = settings.pinecone_index_name

    # List existing indexes
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    # Create index if it does not exist
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment,
            ),
        )

    logger.info(f"Pinecone index ready: {index_name}")
    return pc.Index(index_name)