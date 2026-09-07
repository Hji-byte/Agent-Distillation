# 4096 paired filtering audit

Source: the unchanged 1981 no-think pairs (v3), using the local Qwen3.5-0.8B
tokenizer, complete chat length, enable_thinking=False. A pair is removed if
either answer's full conversation exceeds 4096. No resampling or truncation.

94 pairs excluded; 1887 retained. Compared with the 1940-pair 6400 version,
53 more pairs are excluded. Validation remains byte-identical (188 questions).
Tests verify paired IDs/order, exact source records, threshold membership,
excluded/retained disjointness, and validation preservation.

Difficulty and subject metadata are joined by question ID from
`../cot_teacher_train_v3_2000/train.json`; all retained IDs are matched.

| Difficulty | 6400 | 4096 | Additional exclusions |
|---|---:|---:|---:|
| Level 2 | 435 | 435 | 0 |
| Level 3 | 560 | 556 | 4 |
| Level 4 | 427 | 419 | 8 |
| Level 5 | 518 | 477 | 41 |
| Total | 1940 | 1887 | 53 |

Level 5 share decreases from 26.70% to 25.28%. Length filtering changes the
distribution; it is not random subsampling. Both training arms remain paired.

| Subject | 6400 | 4096 |
|---|---:|---:|
| Algebra | 455 | 454 |
| Counting & Probability | 203 | 194 |
| Geometry | 169 | 160 |
| Intermediate Algebra | 339 | 319 |
| Number Theory | 243 | 239 |
| Prealgebra | 344 | 341 |
| Precalculus | 187 | 180 |

| Full training sequence tokens | Normal | Shortest |
|---|---:|---:|
| Mean | 1174.43 | 988.16 |
| Maximum | 4065 | 3834 |

Use `scripts/training/run_cot_sft_4096.sh` to train this v5 dataset with the
4096 profile (1887 pairs). The separate 6400 entry still selects v4.
Cloud training memory feasibility has not yet been verified.
