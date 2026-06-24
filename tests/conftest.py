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
    from property_intel.db.session import drop_db, get_engine, init_db

    engine = get_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip(f"PostgreSQL not reachable at {TEST_DATABASE_URL}")

    init_db(engine)
    yield engine
    drop_db(engine)
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
