from __future__ import annotations

from cryptography.fernet import Fernet

from app.storage.database import Database, database_engine_options


def test_database_defaults_to_existing_sqlite_file(tmp_path) -> None:
    database = Database(tmp_path)

    assert database.engine.url.get_backend_name() == "sqlite"
    assert database.engine.url.database == str(tmp_path / "aka_multi_agent.db")


def test_explicit_sqlite_database_url_is_honored(tmp_path) -> None:
    target = tmp_path / "shared.db"
    database = Database(tmp_path, database_url=f"sqlite:///{target}")

    assert database.engine.url.database == str(target)


def test_postgresql_engine_options_do_not_include_sqlite_connect_args() -> None:
    options = database_engine_options("postgresql+psycopg://app:secret@db/aka")

    assert options["pool_pre_ping"] is True
    assert "connect_args" not in options


def test_shared_model_secret_key_supports_multiple_instances(tmp_path, monkeypatch) -> None:
    shared_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("AKA_MODEL_SECRET_KEY", shared_key)
    first = Database(tmp_path / "instance-a")
    second = Database(tmp_path / "instance-b")

    encrypted = first.encrypt("provider-secret")

    assert second.decrypt(encrypted) == "provider-secret"
