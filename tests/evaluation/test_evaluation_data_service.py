from unittest.mock import MagicMock

import pytest
from assertpy import assert_that

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.evaluation.evaluation_data_service import EvaluationDataService


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.postgres.connection_string = "sqlite:///:memory:"
    return settings


@pytest.fixture
def evaluation_service(mock_settings):
    service = EvaluationDataService(mock_settings)
    service.initialize_database()
    return service


def test_should_save_question(evaluation_service):
    result = evaluation_service.save_question(
        question="What is the Ring?",
        ground_truth_answer="The One Ring of Power",
        ground_truth_contexts=["Context 1", "Context 2"],
        source_metadata={"source": "lotr.pdf"},
    )

    assert_that(result.id).is_not_none()
    assert_that(result.question).is_equal_to("What is the Ring?")
    assert_that(result.ground_truth_answer).is_equal_to("The One Ring of Power")
    assert_that(result.ground_truth_contexts).is_equal_to(["Context 1", "Context 2"])


def test_should_save_questions_batch(evaluation_service):
    questions = [
        {
            "question": "Question 1",
            "ground_truth_answer": "Answer 1",
            "ground_truth_contexts": ["Context 1"],
        },
        {
            "question": "Question 2",
            "ground_truth_answer": "Answer 2",
            "ground_truth_contexts": ["Context 2"],
        },
    ]

    results = evaluation_service.save_questions_batch(questions)

    assert_that(results).is_length(2)
    assert_that(results[0].question).is_equal_to("Question 1")
    assert_that(results[1].question).is_equal_to("Question 2")


def test_should_get_all_questions(evaluation_service):
    evaluation_service.save_question(
        question="Q1",
        ground_truth_answer="A1",
        ground_truth_contexts=[],
    )
    evaluation_service.save_question(
        question="Q2",
        ground_truth_answer="A2",
        ground_truth_contexts=[],
    )

    results = evaluation_service.get_all_questions()

    assert_that(results).is_length(2)


def test_should_get_question_count(evaluation_service):
    evaluation_service.save_question(
        question="Q1",
        ground_truth_answer="A1",
        ground_truth_contexts=[],
    )

    count = evaluation_service.get_question_count()

    assert_that(count).is_equal_to(1)


def test_should_clear_questions(evaluation_service):
    evaluation_service.save_question(
        question="Q1",
        ground_truth_answer="A1",
        ground_truth_contexts=[],
    )
    evaluation_service.save_question(
        question="Q2",
        ground_truth_answer="A2",
        ground_truth_contexts=[],
    )

    deleted_count = evaluation_service.clear_questions()

    assert_that(deleted_count).is_equal_to(2)
    assert_that(evaluation_service.get_question_count()).is_equal_to(0)


def test_should_return_zero_count_for_empty_database(evaluation_service):
    count = evaluation_service.get_question_count()

    assert_that(count).is_equal_to(0)
