Mean±sd across repeats (per-repeat means). – = metric not applicable to the tasks.

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
