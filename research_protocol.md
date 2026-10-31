# Research protocol

## Objective

The project compares a general-purpose LLM against a specialized scientific literature-analysis workflow for the same uploaded paper and the same research question.

## Research question

Can a system explicitly designed for scientific literature analysis outperform a general-purpose LLM on research-oriented tasks when both are asked the same question over the same paper?

## Conditions

### Pipeline A: General-purpose LLM

This condition does not use specialised retrieval, semantic indexing, or layered research instructions. It sends the uploaded document text and the question to the configured LLM through a simple baseline prompt.

### Pipeline B: Specialized literature-analysis pipeline

This condition uses the same underlying model but adds architecture-specific processing:

- PDF extraction with PyMuPDF
- Preprocessing and text cleaning
- Chunking with page metadata
- Embedding generation with a sentence-transformer model
- Semantic retrieval using FAISS or an equivalent fallback
- Evidence-grounded prompt design
- Structured answer format with uncertainty and limit statements
- Traceable source references from retrieved evidence

## Fair comparison design

The application keeps the comparison fair by using the same:

- uploaded PDF corpus,
- research question,
- model endpoint configuration where possible,
- same underlying LLM provider and model,
- same user evaluation rubric.

## Evaluation dimensions

Each pipeline is manually scored from 1 to 5 on the following dimensions:

1. Accuracy
2. Relevance
3. Evidence / Traceability
4. Completeness
5. Research-gap quality
6. Consistency

Lower and higher values are not automatically interpreted as performance claims. Scores are user-entered manual judgments, not objective scientific validation.

## Procedure

1. Upload the relevant PDF(s).
2. Choose or enter a research question.
3. Run Pipeline A.
4. Run Pipeline B.
5. Compare the generated answers and retrieved evidence.
6. Score each pipeline using the six dimensions.
7. Export the experiment history as CSV or JSON.

## Interpretation

This app is not designed to conclude that one method is universally better. Instead, it is intended to support transparent, evidence-based comparison and teaching about research workflow design and evaluation.

## Limitations

- The app depends on the quality of the uploaded PDF text extraction.
- Semantic retrieval quality depends on the selected embedding model and chunk design.
- Manual evaluation reflects user judgement and may vary across reviewers.
- The project is a prototype and not a substitute for a large formal benchmark.
