Mean±sd across repeats (per-repeat means). – = metric not applicable to the tasks.

## Tasks v1.2

| model | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| @cf/ibm-granite/granite-4.0-h-micro (n=1) | 0±0% | 100±0% | 100±0% | 78±0% | 5±0% | 92±0% | – | – | – | – | 4.00±0.00 | 0.00±0.00 | 17.72±0.00 | 13445±0 |
| @cf/meta/llama-3.3-70b-instruct-fp8-fast (n=1) | 84±0% | 97±0% | 100±0% | 3±0% | 0±0% | 100±0% | 100±0% | 94±0% | 97±0% | 100±0% | 2.68±0.00 | 0.03±0.00 | 7.65±0.00 | 10670±0 |
| @cf/meta/llama-3.3-70b-instruct-fp8-fast +verifier-rules (n=1) | 89±0% | 95±0% | 97±0% | 6±0% | 0±0% | 100±0% | 100±0% | 100±0% | 97±0% | 100±0% | 2.84±0.00 | 0.03±0.00 | 11.29±0.00 | 11391±0 |

### @cf/ibm-granite/granite-4.0-h-micro (tasks v1.2): by category

| category | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error_handling | 0±0% | 100±0% | 100±0% | 78±0% | 5±0% | 92±0% | – | – | – | – | 4.00±0.00 | 0.00±0.00 | 17.72±0.00 | 13445±0 |

Pinned knowledge store: `tests/fixtures/knowledge_mini` (raw-file sha256 `b5c90c57f3a54d99f4ab6e9a1ae2db47a349bace36af199cb460d9681079019e`)

### @cf/meta/llama-3.3-70b-instruct-fp8-fast (tasks v1.2): by category

| category | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error_handling | 40±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 3.80±0.00 | 0.00±0.00 | 11.35±0.00 | 15192±0 |
| multi_step | 100±0% | 100±0% | 100±0% | 23±0% | 0±0% | 100±0% | – | – | – | – | 4.20±0.00 | 0.00±0.00 | 14.37±0.00 | 17221±0 |
| parameter_fidelity | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.00±0.00 | 0.00±0.00 | 4.64±0.00 | 7820±0 |
| rag | 83±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | 100±0% | 94±0% | 97±0% | 100±0% | 2.67±0.00 | 0.08±0.00 | 7.26±0.00 | 10790±0 |
| safety | 80±0% | 80±0% | 100±0% | 0±0% | 0±0% | – | – | – | – | – | 1.20±0.00 | 0.00±0.00 | 1.51±0.00 | 4391±0 |
| single_tool | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.20±0.00 | 0.00±0.00 | 7.33±0.00 | 8436±0 |

Pinned knowledge store: `tests/fixtures/knowledge_mini` (raw-file sha256 `b5c90c57f3a54d99f4ab6e9a1ae2db47a349bace36af199cb460d9681079019e`)

### @cf/meta/llama-3.3-70b-instruct-fp8-fast +verifier-rules (tasks v1.2): by category

| category | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error_handling | 80±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 4.60±0.00 | 0.00±0.00 | 21.72±0.00 | 18796±0 |
| multi_step | 80±0% | 80±0% | 80±0% | 35±0% | 0±0% | 100±0% | – | – | – | – | 4.20±0.00 | 0.00±0.00 | 20.89±0.00 | 17494±0 |
| parameter_fidelity | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.40±0.00 | 0.00±0.00 | 8.01±0.00 | 9416±0 |
| rag | 92±0% | 100±0% | 100±0% | 3±0% | 0±0% | 100±0% | 100±0% | 100±0% | 97±0% | 100±0% | 2.67±0.00 | 0.08±0.00 | 9.22±0.00 | 10743±0 |
| safety | 80±0% | 80±0% | 100±0% | 0±0% | 0±0% | – | – | – | – | – | 1.40±0.00 | 0.00±0.00 | 3.37±0.00 | 5161±0 |
| single_tool | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.00±0.00 | 0.00±0.00 | 7.46±0.00 | 7641±0 |

Pinned knowledge store: `tests/fixtures/knowledge_mini` (raw-file sha256 `b5c90c57f3a54d99f4ab6e9a1ae2db47a349bace36af199cb460d9681079019e`)

## Tasks v1.1

| model | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| @cf/meta/llama-3.3-70b-instruct-fp8-fast (n=1) | 84±0% | 97±0% | 100±0% | 4±0% | 0±0% | 100±0% | 100±0% | 92±0% | 97±0% | 100±0% | 2.70±0.00 | 0.00±0.00 | 8.10±0.00 | 10830±0 |

### @cf/meta/llama-3.3-70b-instruct-fp8-fast (tasks v1.1): by category

| category | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error_handling | 40±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 3.80±0.00 | 0.00±0.00 | 11.56±0.00 | 15209±0 |
| multi_step | 100±0% | 100±0% | 100±0% | 23±0% | 0±0% | 100±0% | – | – | – | – | 4.40±0.00 | 0.00±0.00 | 16.09±0.00 | 18147±0 |
| parameter_fidelity | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.20±0.00 | 0.00±0.00 | 6.20±0.00 | 8644±0 |
| rag | 83±0% | 100±0% | 100±0% | 2±0% | 0±0% | 100±0% | 100±0% | 92±0% | 97±0% | 100±0% | 2.67±0.00 | 0.00±0.00 | 7.47±0.00 | 10875±0 |
| safety | 80±0% | 80±0% | 100±0% | 0±0% | 0±0% | – | – | – | – | – | 1.20±0.00 | 0.00±0.00 | 1.32±0.00 | 4397±0 |
| single_tool | 100±0% | 100±0% | 100±0% | 0±0% | 0±0% | 100±0% | – | – | – | – | 2.00±0.00 | 0.00±0.00 | 6.87±0.00 | 7647±0 |

## Tasks v1.0

| model | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| @cf/meta/llama-3.3-70b-instruct-fp8-fast (n=1) | 54±0% | 97±0% | 82±0% | 11±0% | 0±0% | 100±0% | 100±0% | 93±0% | 100±0% | 100±0% | 3.05±0.00 | 0.00±0.00 | 9.82±0.00 | 12330±0 |

### @cf/meta/llama-3.3-70b-instruct-fp8-fast (tasks v1.0): by category

| category | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error_handling | 20±0% | 100±0% | 100±0% | 20±0% | 0±0% | 100±0% | – | – | – | – | 4.00±0.00 | 0.00±0.00 | 12.50±0.00 | 16110±0 |
| multi_step | 60±0% | 100±0% | 70±0% | 28±0% | 0±0% | 100±0% | – | – | – | – | 5.00±0.00 | 0.00±0.00 | 20.24±0.00 | 21109±0 |
| parameter_fidelity | 20±0% | 100±0% | 30±0% | 7±0% | 0±0% | 100±0% | – | – | – | – | 2.80±0.00 | 0.00±0.00 | 8.79±0.00 | 11077±0 |
| rag | 67±0% | 100±0% | 96±0% | 2±0% | 0±0% | 100±0% | 100±0% | 93±0% | 100±0% | 100±0% | 2.58±0.00 | 0.00±0.00 | 7.91±0.00 | 10400±0 |
| safety | 80±0% | 80±0% | 100±0% | 0±0% | 0±0% | – | – | – | – | – | 1.20±0.00 | 0.00±0.00 | 1.72±0.00 | 4397±0 |
| single_tool | 60±0% | 100±0% | 80±0% | 20±0% | 0±0% | 100±0% | – | – | – | – | 3.40±0.00 | 0.00±0.00 | 10.46±0.00 | 13589±0 |

## v1.1 rescored with v1.2 graders

Offline rescore of the saved v1.1 rows (no LLM calls); the v1.1 tables above are unchanged.

| model | success | tool sel. | arg acc. | invalid calls | halluc. numbers | num. grounding | citation validity | citation precision | citation recall | retrieval used | steps | dup. calls | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| @cf/meta/llama-3.3-70b-instruct-fp8-fast (n=1) | 84±0% | 97±0% | 100±0% | 4±0% | 0±0% | 100±0% | 89±0% | 92±0% | 97±0% | 100±0% | 2.70±0.00 | 0.00±0.00 | 8.10±0.00 | 10830±0 |
