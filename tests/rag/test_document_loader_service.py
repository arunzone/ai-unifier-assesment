from unittest.mock import MagicMock, patch

from assertpy import assert_that
from langchain_core.documents import Document

from ai_unifier_assesment.rag.document_loader_service import DocumentLoaderService


def test_should_initialize_with_default_chunk_settings():
    service = DocumentLoaderService()

    assert_that(service._chunk_size).is_equal_to(500)
    assert_that(service._chunk_overlap).is_equal_to(100)


def test_should_initialize_with_custom_chunk_settings():
    service = DocumentLoaderService(chunk_size=1000, chunk_overlap=200)

    assert_that(service._chunk_size).is_equal_to(1000)
    assert_that(service._chunk_overlap).is_equal_to(200)


def test_should_load_pdf_file():
    service = DocumentLoaderService()

    with patch("ai_unifier_assesment.rag.document_loader_service.PyPDFLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_loader.return_value = mock_instance
        mock_docs = [Document(page_content="test content", metadata={"source": "test.pdf"})]
        mock_instance.load.return_value = mock_docs

        result = service.load_pdf("test.pdf")

        mock_loader.assert_called_once_with("test.pdf")
        assert_that(result).is_equal_to(mock_docs)


def test_should_split_documents_with_configured_settings():
    service = DocumentLoaderService(chunk_size=100, chunk_overlap=20)
    documents = [Document(page_content="a" * 200, metadata={"source": "test.pdf"})]

    with patch("ai_unifier_assesment.rag.document_loader_service.RecursiveCharacterTextSplitter") as mock_splitter:
        mock_instance = MagicMock()
        mock_splitter.return_value = mock_instance
        split_docs = [Document(page_content="a" * 100, metadata={"source": "test.pdf"})]
        mock_instance.split_documents.return_value = split_docs

        result = service.split_documents(documents)

        mock_splitter.assert_called_once_with(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        mock_instance.split_documents.assert_called_once_with(documents)
        assert_that(result).is_equal_to(split_docs)


def test_should_load_and_split_pdf():
    service = DocumentLoaderService()

    with patch.object(service, "load_pdf") as mock_load:
        with patch.object(service, "split_documents") as mock_split:
            mock_docs = [Document(page_content="test", metadata={})]
            mock_load.return_value = mock_docs
            mock_split.return_value = mock_docs

            result = service.load_and_split("test.pdf")

            mock_load.assert_called_once_with("test.pdf")
            mock_split.assert_called_once_with(mock_docs)
            assert_that(result).is_equal_to(mock_docs)


def test_should_load_pdfs_from_directory():
    service = DocumentLoaderService()

    with patch("ai_unifier_assesment.rag.document_loader_service.Path") as mock_path:
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.glob.return_value = [
            MagicMock(__str__=lambda x: "file1.pdf"),
            MagicMock(__str__=lambda x: "file2.pdf"),
        ]

        with patch.object(service, "load_pdf") as mock_load:
            mock_load.return_value = [Document(page_content="test", metadata={})]

            result = service.load_pdfs_from_directory("/test/dir")

            mock_path.assert_called_once_with("/test/dir")
            mock_path_instance.glob.assert_called_once_with("*.pdf")
            assert_that(mock_load.call_count).is_equal_to(2)
            assert_that(len(result)).is_equal_to(2)


def test_should_load_and_split_directory():
    service = DocumentLoaderService()

    with patch.object(service, "load_pdfs_from_directory") as mock_load:
        with patch.object(service, "split_documents") as mock_split:
            mock_docs = [Document(page_content="test", metadata={})]
            mock_load.return_value = mock_docs
            mock_split.return_value = mock_docs

            result = service.load_and_split_directory("/test/dir")

            mock_load.assert_called_once_with("/test/dir")
            mock_split.assert_called_once_with(mock_docs)
            assert_that(result).is_equal_to(mock_docs)
