from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    host: str = "127.0.0.1"
    api_key: str = "change_me"

    max_file_size_mb: int = 50
    max_pages_ocr: int = 500
    job_timeout_seconds: int = 120
    max_concurrent_jobs: int = 5

    tesseract_cmd: str = ""

    chroma_persist_dir: str = "./chroma_data"

    semantic_similarity_threshold: float = 0.92
    simhash_similarity_threshold: float = 0.85

    class Config:
        env_file = ".env"


settings = Settings()
