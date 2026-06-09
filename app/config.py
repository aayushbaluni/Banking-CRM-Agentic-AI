from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str
    openai_model: str = "gpt-4o"
    db_path: str = "app/db/crm.db"
    model_path: str = "app/ml/model.pkl"


settings = Settings()
