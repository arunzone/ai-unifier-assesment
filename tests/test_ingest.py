from unittest.mock import MagicMock, patch

from assertpy import assert_that

from ai_unifier_assesment.ingest import main, create_ingestion_service, ingest_pdf, ingest_directory, show_stats


def test_should_create_ingestion_service_with_settings():
    with patch("ai_unifier_assesment.ingest.get_settings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.rag.chunk_size = 500
        mock_settings_instance.rag.chunk_overlap = 100
        mock_settings.return_value = mock_settings_instance

        with patch("ai_unifier_assesment.ingest.EmbeddingService"):
            with patch("ai_unifier_assesment.ingest.VectorStoreService"):
                with patch("ai_unifier_assesment.ingest.DocumentLoaderService") as mock_loader:
                    with patch("ai_unifier_assesment.ingest.IngestionService"):
                        create_ingestion_service()

                        mock_loader.assert_called_once_with(chunk_size=500, chunk_overlap=100)


def test_should_ingest_pdf_using_collection_from_settings():
    with patch("ai_unifier_assesment.ingest.get_settings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.rag.chunk_size = 500
        mock_settings_instance.rag.chunk_overlap = 100
        mock_settings_instance.chroma.collection_name = "test_collection"
        mock_settings.return_value = mock_settings_instance

        with patch("ai_unifier_assesment.ingest.create_ingestion_service") as mock_create:
            mock_service = MagicMock()
            mock_service.ingest_pdf.return_value = 10
            mock_create.return_value = mock_service

            ingest_pdf("test.pdf")

            mock_service.ingest_pdf.assert_called_once_with("test.pdf", "test_collection")


def test_should_ingest_directory_using_collection_from_settings():
    with patch("ai_unifier_assesment.ingest.get_settings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.rag.chunk_size = 500
        mock_settings_instance.rag.chunk_overlap = 100
        mock_settings_instance.chroma.collection_name = "test_collection"
        mock_settings.return_value = mock_settings_instance

        with patch("ai_unifier_assesment.ingest.create_ingestion_service") as mock_create:
            mock_service = MagicMock()
            mock_service.ingest_directory.return_value = 50
            mock_create.return_value = mock_service

            ingest_directory("/test/dir")

            mock_service.ingest_directory.assert_called_once_with("/test/dir", "test_collection")


def test_should_show_stats_for_collection():
    with patch("ai_unifier_assesment.ingest.get_settings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.chroma.collection_name = "test_collection"
        mock_settings.return_value = mock_settings_instance

        with patch("ai_unifier_assesment.ingest.create_ingestion_service") as mock_create:
            mock_service = MagicMock()
            mock_service.get_collection_stats.return_value = {
                "collection_name": "test_collection",
                "document_count": 100,
            }
            mock_create.return_value = mock_service

            show_stats()

            mock_service.get_collection_stats.assert_called_once_with("test_collection")


def test_main_should_return_zero_for_stats():
    with patch("ai_unifier_assesment.ingest.show_stats"):
        with patch("sys.argv", ["ingest", "--stats"]):
            result = main()
            assert_that(result).is_equal_to(0)


def test_main_should_return_zero_for_pdf():
    with patch("ai_unifier_assesment.ingest.ingest_pdf"):
        with patch("sys.argv", ["ingest", "--pdf", "test.pdf"]):
            result = main()
            assert_that(result).is_equal_to(0)


def test_main_should_return_zero_for_directory():
    with patch("ai_unifier_assesment.ingest.ingest_directory"):
        with patch("sys.argv", ["ingest", "--directory", "/test/dir"]):
            result = main()
            assert_that(result).is_equal_to(0)


def test_main_should_return_one_for_no_args():
    with patch("sys.argv", ["ingest"]):
        result = main()
        assert_that(result).is_equal_to(1)


def test_main_should_return_one_on_exception():
    with patch("ai_unifier_assesment.ingest.ingest_pdf") as mock_ingest:
        mock_ingest.side_effect = Exception("Test error")
        with patch("sys.argv", ["ingest", "--pdf", "test.pdf"]):
            result = main()
            assert_that(result).is_equal_to(1)
