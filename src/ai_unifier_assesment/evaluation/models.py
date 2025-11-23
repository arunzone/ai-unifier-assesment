from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Integer, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class EvaluationQuestion(Base):
    __tablename__ = "evaluation_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    ground_truth_answer = Column(Text, nullable=False)
    ground_truth_contexts = Column(JSON, nullable=False)
    source_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


def create_database_engine(connection_string: str):
    return create_engine(connection_string)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session_factory(engine):
    return sessionmaker(bind=engine)
