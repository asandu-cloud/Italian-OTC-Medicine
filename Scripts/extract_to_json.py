import json
from pathlib import Path
from pypdf import PdfReader   

# Directories (relative, no hard-coded /Users/… paths)
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "medicinali"
JSON_DIR = ROOT / "kb_json"
JSON_DIR.mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    return "\n".join(pages_text)

def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} pdf files")

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"No text extracted from: {pdf_path.name}")
        data = {
            "filename": pdf_path.name,
            "source_path": str(pdf_path.relative_to(ROOT)),
            "text": text,
        }
        json_path = JSON_DIR / (pdf_path.stem + ".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {json_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
