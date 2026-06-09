from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-4o"
    db_path: str = "app/db/crm.db"
    model_path: str = "app/ml/model.pkl"


settings = Settings()
