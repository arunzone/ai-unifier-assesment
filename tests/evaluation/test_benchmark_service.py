from unittest.mock import MagicMock

from assertpy import assert_that
from langchain_core.documents import Document

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.evaluation.benchmark_service import BenchmarkService
from ai_unifier_assesment.evaluation.evaluation_data_service import EvaluationDataService
from ai_unifier_assesment.evaluation.models import EvaluationQuestion
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


def test_should_return_error_when_no_questions():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    evaluation_service.get_all_questions.return_value = []

    service = BenchmarkService(settings, evaluation_service, vector_store_service)
    result = service.run_retrieval_benchmark()

    assert_that(result).contains_key("error")
    assert_that(result["total_questions"]).is_equal_to(0)


def test_should_calculate_accuracy():
    settings = MagicMock(spec=Settings)
    settings.chroma.collection_name = "test_collection"
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    # Create mock questions
    q1 = MagicMock(spec=EvaluationQuestion)
    q1.id = 1
    q1.question = "Test question 1"
    q1.ground_truth_contexts = ["matching content"]

    q2 = MagicMock(spec=EvaluationQuestion)
    q2.id = 2
    q2.question = "Test question 2"
    q2.ground_truth_contexts = ["different content"]

    evaluation_service.get_all_questions.return_value = [q1, q2]

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = [
        [Document(page_content="matching content", metadata={})],
        [Document(page_content="unrelated content", metadata={})],
    ]
    vector_store_service.get_retriever.return_value = mock_retriever

    service = BenchmarkService(settings, evaluation_service, vector_store_service)
    result = service.run_retrieval_benchmark(k=5)

    assert_that(result["total_questions"]).is_equal_to(2)
    assert_that(result["hits"]).is_equal_to(1)
    assert_that(result["accuracy_percent"]).is_equal_to(50.0)


def test_should_calculate_median_retrieval_time():
    settings = MagicMock(spec=Settings)
    settings.chroma.collection_name = "test_collection"
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    q1 = MagicMock(spec=EvaluationQuestion)
    q1.id = 1
    q1.question = "Q1"
    q1.ground_truth_contexts = []

    evaluation_service.get_all_questions.return_value = [q1]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    vector_store_service.get_retriever.return_value = mock_retriever

    service = BenchmarkService(settings, evaluation_service, vector_store_service)
    result = service.run_retrieval_benchmark()

    assert_that(result).contains_key("median_retrieval_time_ms")
    assert_that(result).contains_key("meets_latency_requirement")


def test_should_check_hit_with_exact_match():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    ground_truth = ["the ring was found"]
    retrieved = ["the ring was found in the river"]

    result = service._check_hit(ground_truth, retrieved)

    assert_that(result).is_true()


def test_should_check_hit_with_overlap():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    ground_truth = ["frodo carried the ring to mordor"]
    retrieved = ["frodo carried the one ring all the way to mordor"]

    result = service._check_hit(ground_truth, retrieved)

    assert_that(result).is_true()


def test_should_not_hit_with_no_overlap():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    ground_truth = ["completely different content"]
    retrieved = ["nothing matches here at all"]

    result = service._check_hit(ground_truth, retrieved)

    assert_that(result).is_false()


def test_should_calculate_median_odd():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    result = service._calculate_median([1.0, 3.0, 2.0])

    assert_that(result).is_equal_to(2.0)


def test_should_calculate_median_even():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    result = service._calculate_median([1.0, 2.0, 3.0, 4.0])

    assert_that(result).is_equal_to(2.5)


def test_should_return_zero_for_empty_median():
    settings = MagicMock(spec=Settings)
    evaluation_service = MagicMock(spec=EvaluationDataService)
    vector_store_service = MagicMock(spec=VectorStoreService)

    service = BenchmarkService(settings, evaluation_service, vector_store_service)

    result = service._calculate_median([])

    assert_that(result).is_equal_to(0.0)
