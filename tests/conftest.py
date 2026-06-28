import os

import pytest
from sqlalchemy.exc import OperationalError

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://property_intel:property_intel_dev_password@localhost:5432/property_intel",
)


@pytest.fixture(scope="session")
def db_engine():
    import property_intel.db.models  # registers all ORM models in Base.metadata before create_all
    from property_intel.db.session import get_engine, init_db

    engine = get_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip(f"PostgreSQL not reachable at {TEST_DATABASE_URL}")

    init_db(engine)
    yield engine
    # NOTE: intentionally NOT calling drop_db here — tests share the dev database
    # and drop_db would destroy real application data (documents + chunks).
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    from property_intel.db.models import ChunkModel, DocumentModel
    from property_intel.db.session import get_session_factory

    session = get_session_factory(db_engine)()
    yield session
    session.rollback()
    session.query(ChunkModel).delete()
    session.query(DocumentModel).delete()
    session.commit()
    session.close()


@pytest.fixture(scope="session")
def qdrant_reachable():
    """Session fixture that skips the whole session if Qdrant is not running."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
    except Exception:
        pytest.skip("Qdrant not reachable at localhost:6333")


@pytest.fixture
def qdrant_store(qdrant_reachable):
    """Fresh QdrantStore collection per test for full isolation."""
    from property_intel.retrieval.vector_store import QdrantStore

    store = QdrantStore()
    store._collection = "test_document_chunks"
    store.drop_collection()
    store.ensure_collection()
    yield store
    store.drop_collection()
