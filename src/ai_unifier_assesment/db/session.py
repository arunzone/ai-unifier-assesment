from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ai_unifier_assesment.config import Settings, get_settings


def get_engine(settings: Annotated[Settings, Depends(get_settings)]):
    return create_engine(settings.postgres.connection_string, pool_pre_ping=True)


def get_session_factory(engine=Depends(get_engine)):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session(session_factory=Depends(get_session_factory)) -> Generator[Session, None, None]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
