import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import tempfile

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# New test for document ingestion
def test_ingest_document():
    # Create a dummy text file
    content = "This is a test document for ingestion. It contains some sample text."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(content.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            response = client.post("/api/ingest", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "chunks_processed" in data
        assert data["chunks_processed"] > 0

    finally:
        # Clean up the dummy file
        os.unlink(temp_file_path)

# TODO: Add more tests for the API endpoints
