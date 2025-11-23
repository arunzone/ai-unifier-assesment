from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoaderService:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_pdfs_from_directory(self, directory_path: str) -> list[Document]:
        documents: list[Document] = []
        path = Path(directory_path)
        for pdf_file in path.glob("*.pdf"):
            documents.extend(self.load_pdf(str(pdf_file)))
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def load_and_split(self, file_path: str) -> list[Document]:
        documents = self.load_pdf(file_path)
        return self.split_documents(documents)

    def load_and_split_directory(self, directory_path: str) -> list[Document]:
        documents = self.load_pdfs_from_directory(directory_path)
        return self.split_documents(documents)
