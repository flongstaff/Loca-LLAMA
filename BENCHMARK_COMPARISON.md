# Loca-LLAMA vs sql-benchmark.nicklothian.com — Comparison & Improvement Plan

## What sql-benchmark.nicklothian.com Does (That We Should Learn From)

### 1. Data Source: AdventureWorks (Real-World Business Data)
**Their approach**: Uses Microsoft AdventureWorks sample database — real business data with Sales, Products, Customers, multiple fact/dimension tables, realistic relationships and data volumes.

**Our approach**: Synthetic e-commerce data with 4 tables (customers, products, orders, order_items) generated via Random(42). Simpler schema, smaller scope.

**Gap**: Their data is more realistic and has more complex relationships (multiple fact tables, calculated columns, fiscal year mappings). Our schema is toy-sized by comparison.

**Improvement**: Consider adopting AdventureWorks data (MIT licensed) or creating a richer synthetic schema with 6-8 tables including calculated metrics, multiple join paths, and ambiguous column names that test the model's schema understanding.

---

### 2. Tool Use / Function Calling (Agentic Loop)
**Their approach**: Uses proper OpenAI-style function calling / tool use. The model calls a `run_sql_query` tool, the harness executes it, returns results to the model. The model can inspect results and decide if they're correct before marking "done". This is a TRUE agentic loop — the model drives the iteration.

**Our approach**: We prompt the model to generate SQL in markdown fences, extract it with regex, execute it, and on failure feed the error back in a new prompt. The model doesn't "see" the results of successful queries — we decide pass/fail externally.

**Gap**: Their approach is significantly more realistic for agentic applications. Our regex extraction is brittle compared to structured tool calls.

**Improvement**:
- Add function calling mode for runtimes that support it (LM Studio, oMLX both support OpenAI tool use)
- Let the model inspect result summaries before marking complete
- Keep the current prompt-based mode as fallback for runtimes without tool support

---

### 3. Cost Tracking
**Their approach**: Tracks cost per benchmark run ($0.03 for Grok 4.1 Fast, $0.63 for Opus). Essential for comparing value-for-money.

**Our approach**: We have `extra["cost_cents"]` convention documented but never populated. No cost calculation.

**Gap**: Cost tracking is a key differentiator in their leaderboard.

**Improvement**: Calculate cost from token counts × pricing. For local models, estimate electricity/time cost. For cloud (OpenRouter), use their pricing API.

---

### 4. Time Tracking (Total Wall-Clock Per Run)
**Their approach**: Reports total seconds per model (186s for GLM-5-Turbo, 405s for Grok 4.1 Fast). This is the total time to answer all 25 questions.

**Our approach**: We track per-question timing (speed_tps, ttft_ms, total_ms) but don't prominently surface total benchmark wall-clock time.

**Gap**: Total time is an important UX metric — how long does the user wait?

**Improvement**: Add total_time_seconds to the benchmark report and leaderboard.

---

### 5. Question Difficulty Calibration
**Their approach**:
- **Trivial** (8 questions): Single table, simple SELECT, max 2 columns, no aggregation
- **Easy** (2 questions): 1-2 tables, basic aggregation
- **Medium** (2 questions): 2+ tables, GROUP BY, HAVING
- **Hard** (13 questions): Multiple joins, window functions, subqueries, CTEs, calculated metrics

**Our approach**:
- **Trivial** (4): Single table counts/lists
- **Easy** (5): Basic filters, ORDER BY
- **Medium** (9): JOINs, GROUP BY, subqueries
- **Hard** (7): Window functions, CTEs, self-joins

**Gap**: Their distribution is heavily skewed toward hard (13/25 = 52% hard). This creates much better model separation. Our distribution is more balanced but less discriminating — most models score similarly on easy/medium.

**Improvement**: Add 5-8 more hard questions. Target 60% hard questions for better model separation. Include questions that only 2-4 top models can answer (like their Q9 which only 4/all models get right).

---

### 6. In-Browser Benchmark Execution
**Their approach**: Full DuckDB-WASM execution in the browser. Users paste an endpoint + model + API key and run the exact same benchmark code. Results appear live.

**Our approach**: We have a DuckDB-WASM playground in the SQL report for exploring data, but no in-browser benchmark execution. The benchmark runs server-side via CLI or API.

**Gap**: Their in-browser execution is a major adoption driver — zero setup.

**Improvement**: This is a large effort. Lower-hanging fruit: improve the existing DuckDB playground with pre-loaded questions and expected results for manual comparison.

---

### 7. Multiple Runs Per Model
**Their approach**: Runs each model multiple times (the leaderboard shows "4/4" or "6/6" per question — indicates multi-run consistency).

**Our approach**: Single run per question by default, with retries on failure.

**Gap**: Multiple runs measure consistency/reliability, not just capability.

**Improvement**: Add `--runs N` flag to `loca-llama sql`. Report pass rate as X/N per question. Show consistency in the heatmap.

---

### 8. Result Scoring Methodology
**Their approach**: Score SQL results, not SQL text. "Allowances for rounding to different decimal places, but other than that we expect a match." Three outcomes: Pass, Fail, Error.

**Our approach**: Same philosophy — we score results not SQL. We have 1% relative tolerance for floats, case-insensitive string comparison, column alias tolerance. Three outcomes: pass, fail, error. Plus partial credit (0.5) for row-count match.

**Gap**: Our approach is actually more lenient (partial credit, alias tolerance). This may inflate scores compared to theirs.

**Improvement**: Add a "strict mode" that matches their scoring exactly. Report both lenient and strict scores.

---

## What We Do That They Don't

### 1. Local-First Focus
We auto-detect LM Studio, oMLX, llama.cpp servers and benchmark models running locally. Their benchmark is cloud-API focused (OpenRouter).

### 2. Speed + Quality Combined
We measure tok/s, TTFT, and per-token latency alongside SQL correctness. They track total time but not per-token metrics.

### 3. Unified Scorecard
We combine SQL, speed, quality (coding tasks), eval (GSM8K/ARC/etc), and throughput into one weighted report. They focus solely on SQL generation.

### 4. Hardware Awareness
We factor in Apple Silicon specs, memory constraints, and quantization trade-offs. They don't consider hardware at all.

### 5. Offline/Air-Gapped Operation
Our benchmark works fully offline with local models. Theirs requires internet access to an API endpoint.

---

## What We Currently Test/Benchmark

| Benchmark | What It Tests | Questions/Tasks | Metrics |
|-----------|--------------|-----------------|---------|
| **Speed** (`loca-llama speed`) | Token generation speed | 5 prompt types × N runs | tok/s, TTFT, p50/p90/p95/p99 |
| **Quality** (`loca-llama quality`) | Coding + reasoning | 10 tasks | pass/fail, composite score, code execution |
| **SQL** (`loca-llama sql`) | SQL generation from NL | 25 questions, 4 tiers | pass/fail/error, retries, per-question timing |
| **Eval** (`loca-llama eval`) | Standard LLM evals | GSM8K, ARC, HellaSwag, IFEval, HumanEval, MMLU | accuracy, errors tracked separately |
| **Throughput** (`loca-llama throughput`) | Concurrent load | Configurable concurrency | tok/s, p50/p90/p99 latency |
| **Unified Report** (`loca-llama report`) | All of the above | Combined scorecard | Weighted overall score |

---

## Priority Improvements (Ordered by Impact)

### P0 — High Impact, Moderate Effort

1. **Add tool use / function calling mode to SQL benchmark**
   - Use OpenAI-compatible tool definitions for `run_sql_query`
   - Model drives the agentic loop, not just regex extraction
   - Keep prompt-based mode as fallback

2. **Add 8 more hard questions to close the difficulty gap**
   - Multi-table revenue analysis with calculated metrics
   - Fiscal year rollups
   - Percentage-of-total calculations
   - Year-over-year comparisons
   - Cross-channel comparisons

3. **Add cost tracking to benchmark results**
   - Calculate from input/output token counts × known pricing
   - Display in leaderboard and reports

### P1 — Medium Impact, Low Effort

4. **Add `--runs N` for multi-run consistency**
   - Report X/N pass rate per question
   - Show consistency heatmap

5. **Add total wall-clock time to SQL benchmark reports**
   - Already have per-question timing — just sum it

6. **Add strict scoring mode**
   - Match sql-benchmark.nicklothian.com's exact scoring
   - Disable partial credit and alias tolerance

7. **Richer schema (6-8 tables)**
   - Add fiscal calendar, sales territories, product subcategories
   - More realistic data relationships

### P2 — Nice to Have

8. **In-browser benchmark execution via DuckDB-WASM**
   - Already have the playground infrastructure
   - Would need TypeScript benchmark harness

9. **AdventureWorks data option**
   - Offer as alternative dataset alongside current synthetic data
   - Enables direct score comparison with sql-benchmark.nicklothian.com

10. **OpenRouter integration for cloud model benchmarking**
    - Direct comparison: local vs cloud scores on same questions
