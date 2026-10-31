from __future__ import annotations

import re
from typing import Any, Dict, List

import fitz


def clean_text_block(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_documents(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        document: Dict[str, Any] = {
            "document_id": f"D{file_index}",
            "filename": getattr(uploaded_file, "name", f"document_{file_index}.pdf"),
            "page_count": 0,
            "status": "uploaded",
            "pages": [],
            "chunks": [],
            "full_text": "",
            "error": "",
        }
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as pdf_document:
                document["page_count"] = pdf_document.page_count
                for page_number in range(pdf_document.page_count):
                    page = pdf_document[page_number]
                    raw_text = page.get_text("text")
                    cleaned_page_text = clean_text_block(raw_text)
                    if not cleaned_page_text:
                        continue
                    document["pages"].append({
                        "page_number": page_number + 1,
                        "text": cleaned_page_text,
                    })
                document["full_text"] = "\n\n".join(page["text"] for page in document["pages"])
        except Exception as exc:  # pragma: no cover - depends on uploaded content
            document["status"] = "error"
            document["error"] = f"Could not open PDF: {exc}"
            documents.append(document)
            continue

        if not document["pages"]:
            document["status"] = "warning"
            document["error"] = "No extractable text was found. This PDF may be scanned or image-based; OCR would be needed."

        chunk_id = 0
        chunk_size = 700
        overlap = 120
        for page in document["pages"]:
            words = re.findall(r"\S+", page["text"])
            if not words:
                continue
            index = 0
            while index < len(words):
                end = min(index + chunk_size, len(words))
                chunk_text = " ".join(words[index:end]).strip()
                if len(chunk_text) >= 80:
                    chunk_id += 1
                    document["chunks"].append({
                        "chunk_id": chunk_id,
                        "source_name": document["filename"],
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    })
                index = end - overlap
                if index <= 0:
                    index = end
        documents.append(document)
    return documents
