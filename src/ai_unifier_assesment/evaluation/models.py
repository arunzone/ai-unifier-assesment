from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, create_engine
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


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, unique=True)
    top_k = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    hits = Column(Integer, nullable=False)
    accuracy_percent = Column(Float, nullable=False)
    median_retrieval_time_ms = Column(Float, nullable=False)
    avg_retrieval_time_ms = Column(Float, nullable=False)
    min_retrieval_time_ms = Column(Float, nullable=False)
    max_retrieval_time_ms = Column(Float, nullable=False)
    meets_latency_requirement = Column(Integer, nullable=False)  # 1 or 0
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class ChatMetrics(Base):
    __tablename__ = "chat_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


def create_database_engine(connection_string: str):
    return create_engine(connection_string)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session_factory(engine):
    return sessionmaker(bind=engine)
