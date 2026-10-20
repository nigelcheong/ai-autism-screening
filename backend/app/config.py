from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/autism_screening"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    supabase_url: str = ""
    supabase_service_key: str = ""
    hf_model_token: str = ""
    model_repo_id: str = "your-username/autism-screening-model"
    # Questionnaire scoring uses the real trained logistic-regression models
    # (app/ml/models/) by default. Image scoring stays stubbed until the
    # PyTorch CNN model is reviewed and wired in.
    use_ml_stub_questionnaire: bool = False
    use_ml_stub_image: bool = False
    # Local disk storage for uploaded screening images (dev only; swap for
    # Supabase Storage in production).
    upload_dir: str = "uploads/images"
    max_image_size_mb: int = 10
    cors_origins: str = "http://localhost:5173"
    # Fernet key encrypting question_responses at rest. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = ""

    class Config:
        env_file = ".env"
        protected_namespaces = ("settings_",)


settings = Settings()
