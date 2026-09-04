from __future__ import annotations

from app.knowledge.repository import KnowledgeRepository
from app.storage.database import Database


def _write_single_chunk_version(repository, dataset_id, document_id, version, text):
    repository.replace_chunks(
        document_id=document_id,
        dataset_id=dataset_id,
        index_version=version,
        parents=[{
            "local_id": "p1", "title": text, "text": text,
            "page_start": 1, "page_end": 1, "token_count": len(text),
        }],
        children=[{
            "local_id": "c1", "parent_local_id": "p1", "title": text,
            "text": text, "page_start": 1, "page_end": 1, "token_count": len(text),
        }],
    )


def test_file_can_link_to_multiple_datasets_and_publish_versions(tmp_path) -> None:
    repository = KnowledgeRepository(Database(tmp_path))
    stored = repository.create_file(
        original_name="manual.txt",
        content_hash="abc123",
        mime_type="text/plain",
        size_bytes=12,
        storage_path="objects/ab/abc123.txt",
    )
    first = repository.create_dataset("售后知识库", parser_profile="manual")
    second = repository.create_dataset("培训知识库", parser_profile="general")

    first_ref = repository.link_file(first.id, stored.id, "manual")
    repository.link_file(second.id, stored.id, "general")

    assert len(repository.list_document_refs(file_id=stored.id)) == 2
    assert first_ref.dataset_id == first.id
    assert repository.get_dataset(first.id).published_version is None


def test_content_hash_is_unique(tmp_path) -> None:
    repository = KnowledgeRepository(Database(tmp_path))
    first = repository.create_file("a.txt", "same", "text/plain", 4, "objects/sa/same.txt")
    second = repository.create_file("b.txt", "same", "text/plain", 4, "objects/sa/same.txt")

    assert first.id == second.id
    assert repository.list_files()[0].original_name == "a.txt"


def test_publishing_second_document_keeps_first_document_online(tmp_path) -> None:
    repository = KnowledgeRepository(Database(tmp_path))
    dataset = repository.create_dataset("多文档知识库")
    first_file = repository.create_file("a.txt", "hash-a", "text/plain", 1, "a.txt")
    second_file = repository.create_file("b.txt", "hash-b", "text/plain", 1, "b.txt")
    first = repository.link_file(dataset.id, first_file.id)
    second = repository.link_file(dataset.id, second_file.id)

    _write_single_chunk_version(repository, dataset.id, first.id, "v-a", "第一本文档")
    repository.publish_dataset(dataset.id, "v-a")
    _write_single_chunk_version(repository, dataset.id, second.id, "v-b", "第二本文档")

    assert {item.text for item in repository.list_children(published_only=True)} == {"第一本文档"}

    repository.publish_dataset(dataset.id, "v-b")

    assert {item.text for item in repository.list_children(published_only=True)} == {"第一本文档", "第二本文档"}
    assert repository.dataset_metrics(dataset.id)["child_count"] == 2


def test_chunk_edit_creates_complete_document_draft_and_is_atomic_on_publish(tmp_path) -> None:
    repository = KnowledgeRepository(Database(tmp_path))
    dataset = repository.create_dataset("可编辑知识库")
    file = repository.create_file("manual.txt", "edit-hash", "text/plain", 1, "manual.txt")
    document = repository.link_file(dataset.id, file.id)
    repository.replace_chunks(
        document_id=document.id,
        dataset_id=dataset.id,
        index_version="v1",
        parents=[{"local_id": "p1", "title": "维护", "text": "旧步骤。保留步骤。", "token_count": 9}],
        children=[
            {"local_id": "c1", "parent_local_id": "p1", "title": "旧步骤", "text": "旧步骤。", "token_count": 4},
            {"local_id": "c2", "parent_local_id": "p1", "title": "保留步骤", "text": "保留步骤。", "token_count": 5},
        ],
    )
    repository.publish_dataset(dataset.id, "v1")
    source = next(item for item in repository.list_children(published_only=True) if item.local_id == "c1")

    edited = repository.edit_child(source.id, text="新步骤。")

    assert edited.index_version.startswith("draft-")
    assert {item.text for item in repository.list_children(published_only=True)} == {"旧步骤。", "保留步骤。"}
    assert {item.text for item in repository.list_children(document_id=document.id, index_version=edited.index_version)} == {
        "新步骤。",
        "保留步骤。",
    }

    repository.publish_dataset(dataset.id, edited.index_version)

    assert {item.text for item in repository.list_children(published_only=True)} == {"新步骤。", "保留步骤。"}
