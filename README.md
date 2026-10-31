# ResearchGapAI

ResearchGapAI is a Streamlit-based research prototype that compares two AI execution conditions for scientific literature analysis:

- Pipeline A: general-purpose LLM using the uploaded PDF text and a simple baseline prompt.
- Pipeline B: specialized scientific literature-analysis workflow with preprocessing, chunking, evidence retrieval, and structured evidence-grounded prompting.

The project is designed for experimental comparison and teaching. It does not train a new model and does not claim that one pipeline is necessarily superior.

## Project overview

This application lets a user upload one or more scientific PDFs, ask a research question, and compare the output of two pipelines:

1. General-purpose LLM pipeline
   - Reads the uploaded document text.
   - Sends the research question and document content to the configured model.
   - Uses a simple baseline prompt without retrieval or specialised evidence handling.

2. Specialized literature-analysis pipeline
   - Extracts text from uploaded PDFs with PyMuPDF.
   - Cleans and chunks the document content.
   - Uses a sentence-transformer embeddings model plus FAISS-based semantic retrieval.
   - Builds a structured evidence-grounded prompt.
   - Returns a literature-analysis answer with evidence references and uncertainty handling.

Both pipelines can be run against the same uploaded PDF and the same research question.

## Technologies used

- Python 3.11+
- Streamlit
- PyMuPDF
- pandas
- scikit-learn (fallback retrieval)
- OpenAI-compatible API client
- sentence-transformers
- FAISS

## Environment variables

Create a local `.env` file from the template and place your values there:

```text
LLM_API_KEY=PASTE_YOUR_KEY_HERE
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=YOUR_AVAILABLE_MODEL_ID
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The app will read these values automatically when run from the project folder.

## Installation on Windows / PowerShell

```powershell
cd "C:\path\to\Research"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the virtual environment activation, you can start Python directly using the venv executable:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

```powershell
cd "C:\path\to\Research"
python -m streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## Using the app

1. Upload one or more PDF files.
2. Select or type a research question.
3. Configure the API settings in the sidebar.
4. Run either pipeline or run both together.
5. Review the answer, evidence, and paper references.
6. Manually score each pipeline using the six evaluation dimensions.
7. Export experiment records to CSV and JSON.

## Evaluation dimensions

The manual evaluation section covers six dimensions:

- Accuracy
- Relevance
- Evidence / Traceability
- Completeness
- Research-gap quality
- Consistency

These are manual user scores and are not scientific validation results.

## Known limitations

- The app relies on uploaded PDF text extraction quality.
- It does not guarantee that a model output is factually correct.
- The specialized pipeline is evidence-grounded, but it still depends on the uploaded document and the chosen model.
- Live model execution requires valid API credentials and provider availability.
- For scanned-only PDFs, OCR would be required for reliable extraction.

## Exported results

CSV and JSON files are saved to the `results/` folder.

## Notes

This project is a research prototype for teaching, comparative analysis, and reproducible experimental design. It intentionally avoids claiming that a new model has been trained or that one pipeline is definitively superior.
