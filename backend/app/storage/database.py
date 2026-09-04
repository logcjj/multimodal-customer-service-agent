from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine


def database_engine_options(database_url: str) -> dict[str, object]:
    options: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False}
    return options


class ModelRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    provider: str
    name: str
    kind: str = Field(index=True)
    base_url: str
    encrypted_secret: str | None = None
    secret_hint: str | None = None
    capabilities_json: str = "[]"
    enabled: bool = True
    is_default: bool = False
    health: str = "untested"
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def capabilities(self) -> list[str]:
        value = json.loads(self.capabilities_json)
        return [str(item) for item in value]


class Database:
    def __init__(self, data_dir: Path, database_url: str | None = None) -> None:
        data_dir = data_dir.expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        resolved_url = database_url or f"sqlite:///{data_dir / 'aka_multi_agent.db'}"
        self.engine = create_engine(resolved_url, **database_engine_options(resolved_url))
        SQLModel.metadata.create_all(self.engine)
        if self.engine.dialect.name == "sqlite":
            self._apply_compatible_migrations()
        self._fernet = Fernet(self._load_or_create_key(data_dir / ".model_secret.key"))

    def _apply_compatible_migrations(self) -> None:
        """Apply additive SQLite migrations needed by earlier V3.1 snapshots."""
        with self.engine.begin() as connection:
            tables = {
                row[0]
                for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            }
            if "tracerecord" in tables:
                columns = {row[1] for row in connection.execute(text("PRAGMA table_info(tracerecord)"))}
                if "spans_json" not in columns:
                    connection.execute(text("ALTER TABLE tracerecord ADD COLUMN spans_json VARCHAR NOT NULL DEFAULT '[]'"))
                if "owner_id" not in columns:
                    connection.execute(
                        text(
                            "ALTER TABLE tracerecord ADD COLUMN owner_id VARCHAR "
                            "NOT NULL DEFAULT '__legacy_anonymous__'"
                        )
                    )
                    connection.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS ix_tracerecord_owner_id "
                            "ON tracerecord (owner_id)"
                        )
                    )
            if "documentrefrecord" in tables:
                columns = {row[1] for row in connection.execute(text("PRAGMA table_info(documentrefrecord)"))}
                if "published_version" not in columns:
                    connection.execute(text("ALTER TABLE documentrefrecord ADD COLUMN published_version VARCHAR"))
                    connection.execute(
                        text(
                            "UPDATE documentrefrecord SET published_version = active_version "
                            "WHERE active_version IS NOT NULL AND dataset_id IN "
                            "(SELECT id FROM datasetrecord WHERE published_version IS NOT NULL)"
                        )
                    )

    @staticmethod
    def _load_or_create_key(path: Path) -> bytes:
        shared = os.getenv("AKA_MODEL_SECRET_KEY", "").strip()
        if shared:
            key = shared.encode("ascii")
            Fernet(key)
            return key
        if path.exists():
            return path.read_bytes().strip()
        key = Fernet.generate_key()
        path.write_bytes(key)
        path.chmod(0o600)
        return key

    def session(self) -> Session:
        return Session(self.engine)

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
