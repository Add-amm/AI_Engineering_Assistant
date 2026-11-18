# Retrieval and answer evaluation

Run the commands in the [README](../README.md) to reproduce this comparison. The frozen benchmark has 60 questions, including 54 answerable cases and six out-of-scope cases. Four require passages from two distinct sources. Each answerable case includes a reference answer and quoted passage anchors. [Benchmark](../corpus/evaluation.json)

Retrieval metrics average over answerable questions. Recall is the fraction of distinct annotated evidence groups found in the first five passages. MRR uses the rank of the first relevant passage. nDCG discounts lower ranks and counts the first match to each evidence group. The latency columns cover search for all 60 questions and exclude index construction and generation.

## Retrieval results

Measured with Python 3.12.14 on Linux x86-64 using TF-IDF plus LSA, FAISS, and the local token-and-phrase reranker. Both strategies targeted 360 lexical pieces per chunk.

| Chunking | Method | Recall@5 | MRR@5 | nDCG@5 | P50 ms | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Headings · 961 passages | BM25 | 0.7407 | 0.5096 | 0.5629 | 1.44 | 2.68 |
| Headings | Dense | 0.5278 | 0.2957 | 0.3413 | 2.14 | 2.79 |
| Headings | Hybrid RRF | 0.6759 | 0.3926 | 0.4527 | 3.44 | 4.12 |
| Headings | Hybrid + rerank | 0.7685 | 0.4963 | 0.5619 | 3.87 | 5.22 |
| Recursive · 502 passages | BM25 | 0.7222 | 0.4552 | 0.5099 | 1.05 | 1.54 |
| Recursive | Dense | 0.7778 | 0.4676 | 0.5353 | 1.85 | 2.65 |
| Recursive | Hybrid RRF | 0.8241 | 0.5198 | 0.5853 | 2.60 | 3.48 |
| Recursive | Hybrid + rerank | 0.6944 | 0.4256 | 0.4851 | 3.04 | 3.63 |

For heading-aware passages, BM25 ranks the first relevant result highest on average. Reranking gives the best Recall@5 within that chunking strategy. Recursive chunks with plain hybrid search produce the best overall retrieval scores in this run. The application uses heading-aware reranking for readable section-level citations and cross-topic evidence. The longer recursive passages often cover more annotated material with a single result. The reranker and chunking strategy should be evaluated together when the corpus changes.

## Local answers

Heading-aware reranking with extractive generation answered 52 of 54 in-scope questions and abstained on all six out-of-scope questions. Overall abstention accuracy was 58 out of 60, or 96.7%. The answers cited 40.7% of the annotated evidence groups on average. This measures overlap with fixed anchors and can miss alternative supporting passages. The scores describe this benchmark and do not establish general factual accuracy.

| Case | Manual review |
| --- | --- |
| `fa-01` · How do FastAPI dependencies work? | Correct. The linked section says that FastAPI executes a declared dependency and injects its result. |
| `tr-01` · How does AutoTokenizer turn text into model inputs? | Partially correct. The cited text explains the input IDs and attention mask, but skips initializing the tokenizer with `AutoTokenizer.from_pretrained`. |
| `cross-01` · Sentence Transformers search in a FastAPI endpoint | Partial. The response links one FastAPI page and a Sentence Transformers page. It offers separate grounding for routing and embeddings, with the integrated implementation still left to the reader. |
| `cross-04` · BERT vs GPT-2 pretraining | Partial. It explains BERT's bidirectionality and masked tokens, while the GPT-2 excerpt discusses inference caching instead of next-token training. |
| `fa-06` · Restrict a path segment to known strings | False abstention. The official route-parameter page contains the relevant Enum example. |
| `out-01` · Tomorrow's weather in Paris | Appropriate abstention. The static corpus has no live forecast evidence. |

The extractive path quotes passages, so the main failure mode is selecting true but incomplete or off-target text. These manual checks assess the cited excerpts against the questions rather than using a model judge. Live model answer correctness and faithfulness remain unmeasured. The OpenAI-compatible and Ollama adapters passed request and response tests with test doubles. The neural retrieval adapters ran with small local test models. Pretrained weights and live LLM endpoints require model access in the deployment environment.

The corpus is an English snapshot of selected upstream documentation. Broader coverage, multilingual retrieval, and model-backed synthesis are separate experiments. The source inventory retains the corresponding licenses and exact revisions. [Source manifest](../corpus/manifest.json)
