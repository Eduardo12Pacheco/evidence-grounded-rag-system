# Product — evidence-grounded-rag-system

Status: **scaffold / implementation not started**

## Platform

Web (local demo) + offline evaluation pipeline.

## Users

- Evaluators who must decide whether a RAG answer can be trusted.
- Researchers and engineers who want a reproducible faithfulness benchmark.
- The author: demonstrating retrieval, embeddings, and evidence discipline.

## Problem

Fluent answers are not trustworthy answers. Without mandatory citation and
abstention, a RAG system can sound certain while fabricating. This product
makes evidence the product: answers are only as good as the passages that
support them, and the system says "I don't know" when evidence is missing.

## What it is NOT yet

- Not a search engine, no API, no production deployment.
- Not using any real corpus (none is licensed yet).
- Not claiming any accuracy number.

## Success metric (product)

A reviewer can ask a question, see the cited passages, verify each citation,
and observe a clean abstention — in under a minute, without reading code.

## Risks

- Corpus access: nothing runs on real documents until licenses are confirmed.
- Faithfulness is hard to measure well → protocol committed before results.

## No-goals

- No multi-tenant SaaS, no user accounts.
- No mobile app.
- No unlicensed corpus indexing.

## Definition of done

- [ ] Demo answers "can I trust this answer and why" with citations in a minute.
- [ ] Every reported number is reproducible with one command.
- [ ] Abstention and citation limits are tested and documented.
