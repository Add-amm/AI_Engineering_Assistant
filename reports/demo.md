# DocLens demo

This walkthrough fits a 15-minute presentation. Start the app with the command in the [README](../README.md).

| Time | Demo |
| --- | --- |
| 0–2 min | Open the assistant. Explain why package versions and line-level citations matter for engineering decisions. |
| 2–5 min | Ask “How do FastAPI dependencies work?” Follow a citation to the frozen source and inspect the retrieved passages. |
| 5–8 min | Ask “How can I build a FastAPI endpoint that performs semantic search using Sentence Transformers?” Inspect evidence from both topics. Show where the extractive answer stops short of an integrated implementation. |
| 8–10 min | Ask “Will it rain in Paris tomorrow?” Show the abstention and inspect any weak matches. |
| 10–12 min | In the retrieval lab, search `HTTPException status code` with BM25 and then with reranking. Compare source sections rather than raw scores across methods. Open the source library to show the pinned FastAPI revision. |
| 12–15 min | Compare the two chunking strategies and four retrieval modes using the measured results. Discuss out-of-scope questions, the limits of excerpts, and how to rerun with local or hosted models. |

[Evaluation report](evaluation.md)
