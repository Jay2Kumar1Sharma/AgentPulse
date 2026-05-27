from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import get_settings


settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    if settings.database_url.startswith("sqlite"):
        _ensure_sqlite_columns()


def _ensure_sqlite_columns() -> None:
    with engine.begin() as connection:
        table_names = {
            row[0]
            for row in connection.execute(text("select name from sqlite_master where type='table'")).fetchall()
        }
        if "evaluation_results" not in table_names:
            return

        existing_columns = {
            row[1]
            for row in connection.execute(text("pragma table_info(evaluation_results)")).fetchall()
        }
        if "llm_judge_result_json" not in existing_columns:
            connection.execute(text("alter table evaluation_results add column llm_judge_result_json TEXT"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
