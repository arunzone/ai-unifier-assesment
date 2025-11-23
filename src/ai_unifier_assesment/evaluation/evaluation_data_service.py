from typing import Annotated, cast

from fastapi import Depends
from sqlalchemy.orm import Session

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.evaluation.models import (
    EvaluationQuestion,
    create_database_engine,
    create_tables,
    get_session_factory,
)


class EvaluationDataService:
    def __init__(self, settings: Annotated[Settings, Depends(get_cached_settings)]):
        self._settings = settings
        self._engine = create_database_engine(settings.postgres.connection_string)
        self._session_factory = get_session_factory(self._engine)

    def initialize_database(self) -> None:
        create_tables(self._engine)

    def get_session(self) -> Session:
        return cast(Session, self._session_factory())

    def save_question(
        self,
        question: str,
        ground_truth_answer: str,
        ground_truth_contexts: list[str],
        source_metadata: dict | None = None,
    ) -> EvaluationQuestion:
        session = self.get_session()
        try:
            eval_question = EvaluationQuestion(
                question=question,
                ground_truth_answer=ground_truth_answer,
                ground_truth_contexts=ground_truth_contexts,
                source_metadata=source_metadata,
            )
            session.add(eval_question)
            session.commit()
            session.refresh(eval_question)
            return eval_question
        finally:
            session.close()

    def save_questions_batch(
        self,
        questions: list[dict],
    ) -> list[EvaluationQuestion]:
        session = self.get_session()
        try:
            eval_questions = []
            for q in questions:
                eval_question = EvaluationQuestion(
                    question=q["question"],
                    ground_truth_answer=q["ground_truth_answer"],
                    ground_truth_contexts=q["ground_truth_contexts"],
                    source_metadata=q.get("source_metadata"),
                )
                session.add(eval_question)
                eval_questions.append(eval_question)
            session.commit()
            for eq in eval_questions:
                session.refresh(eq)
            return eval_questions
        finally:
            session.close()

    def get_all_questions(self) -> list[EvaluationQuestion]:
        session = self.get_session()
        try:
            return cast(list[EvaluationQuestion], session.query(EvaluationQuestion).all())
        finally:
            session.close()

    def get_question_count(self) -> int:
        session = self.get_session()
        try:
            return cast(int, session.query(EvaluationQuestion).count())
        finally:
            session.close()

    def clear_questions(self) -> int:
        session = self.get_session()
        try:
            count = session.query(EvaluationQuestion).delete()
            session.commit()
            return cast(int, count)
        finally:
            session.close()
