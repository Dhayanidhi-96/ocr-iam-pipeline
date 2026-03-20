# Data Audit Report - IAM Handwriting Dataset

## Dataset Overview

| Split | Total | Empty | Very Short | Avg Length | Duplicates |
|---|---|---|---|---|---|
| train | 6,482 | 0 | 1 | 43.4 | 295 |
| validation | 976 | 0 | 0 | 43.1 | 0 |
| test | 2,915 | 0 | 0 | 43.1 | 2 |

## Normalization Decisions

| Decision | Choice | Reason |
|---|---|---|
| Case | Preserve | Real handwriting has capitals. TrOCR trained with case. |
| Punctuation | Keep | Part of real sentences. Removing hurts model. |
| Digits | Keep | Rare but valid real-world content. |
| Whitespace | Strip | Clean hygiene, no information lost. |
| Samples <= 2 chars | Remove | No learning value, likely labelling errors. |
| Duplicates | Keep | Different images, same text = valid diversity. |
