from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from property_intel.config import get_settings
from property_intel.db.base import Base


def get_engine(database_url: str | None = None, connect_timeout: int = 5) -> Engine:
    url = database_url or get_settings().database_url
    return create_engine(url, future=True, connect_args={"connect_timeout": connect_timeout})


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def drop_db(engine: Engine) -> None:
    Base.metadata.drop_all(engine)
