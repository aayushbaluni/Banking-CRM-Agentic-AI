from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"
    db_path: str = "app/db/crm.db"
    model_path: str = "app/ml/model.pkl"


settings = Settings()
