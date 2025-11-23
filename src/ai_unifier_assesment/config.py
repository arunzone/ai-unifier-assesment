from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FastAPIConfig(BaseModel):
    port: int = 8000
    host: str = "0.0.0.0"  # nosec B104 - required for Docker


class OpenAIConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str = "Gpt4o"


class PricingConfig(BaseModel):
    input_cost_per_1m: float
    output_cost_per_1m: float


class OllamaConfig(BaseModel):
    base_url: str
    embedding_model: str


class ChromaConfig(BaseModel):
    host: str
    port: int
    collection_name: str


class RAGConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Flat env vars mapped to fields
    openai_base_url: str = Field(alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    model_name: str = Field(default="Gpt4o", alias="MODEL_NAME")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")
    fastapi_host: str = Field(default="0.0.0.0", alias="FASTAPI_HOST")  # nosec B104
    pricing_input_cost_per_1m: float = Field(default=2.50, alias="PRICING_INPUT_COST_PER_1M")
    pricing_output_cost_per_1m: float = Field(default=10.00, alias="PRICING_OUTPUT_COST_PER_1M")
    memory_window_size: int = Field(default=5, alias="MEMORY_WINDOW_SIZE")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection_name: str = Field(default="rag_corpus", alias="CHROMA_COLLECTION_NAME")
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=100, alias="RAG_CHUNK_OVERLAP")

    @property
    def openai(self) -> OpenAIConfig:
        return OpenAIConfig(
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
            model_name=self.model_name,
        )

    @property
    def fastapi(self) -> FastAPIConfig:
        return FastAPIConfig(
            port=self.fastapi_port,
            host=self.fastapi_host,
        )

    @property
    def pricing(self) -> PricingConfig:
        return PricingConfig(
            input_cost_per_1m=self.pricing_input_cost_per_1m,
            output_cost_per_1m=self.pricing_output_cost_per_1m,
        )

    @property
    def ollama(self) -> OllamaConfig:
        return OllamaConfig(base_url=self.ollama_base_url, embedding_model=self.ollama_embedding_model)

    @property
    def chroma(self) -> ChromaConfig:
        return ChromaConfig(
            host=self.chroma_host,
            port=self.chroma_port,
            collection_name=self.chroma_collection_name,
        )

    @property
    def rag(self) -> RAGConfig:
        return RAGConfig(
            chunk_size=self.rag_chunk_size,
            chunk_overlap=self.rag_chunk_overlap,
        )


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
