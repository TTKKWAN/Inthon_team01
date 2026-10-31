# claude_val: validation sets checked against the public leaderboard

All sets use only the 626 valid episodes of `baseline_v2/validation/splits.json`.
Goal: a set where our submissions rank like the public LB
(starter 0.350 < minimal 0.426 / marked_v1 0.413 / v2 0.439 < hyuwooma centrality 0.52).

| set | negatives | source |
|---|---|---|
| `claude_short` | 4 single edits of the first-sentence GT | **written by Claude** (`parts/`), validation only, user-approved |
| `claude_ab_ac` | {a, b, a+b, a+c} from the edits above, merged by `diff_probe/sim_valsets.merge_edits` | code over `parts/` |
| `qwen_full/short` | 4 Qwen3-4B single edits of the GT (team01 `merged/bank_qval.json`) | HF `SANGchu-03/INThon_Augmented` |
| `para_negs_full/short` | GT positive, 4 Qwen edits of Qwen paraphrases of the GT (structure from session inthon-0e) | HF `qwen/qwen_paraphrases_and_negatives.json` |
| `af_full` | copy of `baseline_v2/validation/val_af_full.json` | rule negatives |

`parts/` is Claude-written text. **Never use it, or any set built from it, for training or
augmentation**; the competition forbids Claude-generated training data. It exists only to score models.

    # HF files go in claude_val/hf/ (not committed): merged/bank_qval.json, qwen/*.json
    python claude_val/build_sets.py
    python claude_val/eval_subs.py --sets claude_val/sets/*.json --centrality
    python claude_val/eval_subs.py --sets claude_val/sets/*.json --sub v2=baseline_v2/outputs/v2_submission_int8

## Result (2026-10-31): no set or mix reproduces the LB order

LB: starter 0.350 < marked_v1 0.413 < minimal 0.426 < v2 0.439 < centrality 0.52.

| set | starter | marked_v1 | minimal | v2int8 | centrality byte | span |
|---|---|---|---|---|---|---|
| af_full | 0.352 | 0.883 | 0.407 | 0.635 | 0.996 | 0.985 |
| claude_short | 0.409 | 0.803 | 0.427 | 0.500 | 0.998 | 0.970 |
| claude_ab_ac | 0.496 | 0.320 | 0.501 | 0.612 | 0.348 | 0.345 |
| qwen_full | 0.408 | 0.748 | 0.389 | 0.488 | 0.994 | 0.972 |
| qwen_short | 0.410 | 0.855 | 0.560 | 0.561 | 0.996 | 0.991 |
| qwen_share3_full | 0.541 | 0.357 | 0.545 | 0.651 | 0.219 | 0.227 |
| qwen_share3_short | 0.520 | 0.443 | 0.709 | 0.733 | 0.211 | 0.153 |
| para_negs_full | 0.435 | 0.565 | 0.313 | 0.716 | 0.501 | 0.214 |
| para_negs_short | 0.438 | 0.510 | 0.552 | 0.680 | 0.580 | 0.333 |
| para_same_full | 0.412 | 0.736 | 0.406 | 0.496 | 0.974 | 0.940 |
| para_same_short | 0.462 | 0.831 | 0.539 | 0.577 | 0.981 | 0.962 |
| para_share3_full | 0.531 | 0.311 | 0.570 | 0.599 | 0.242 | 0.221 |
| para_share3_short | 0.532 | 0.362 | 0.669 | 0.722 | 0.245 | 0.155 |

- One-edit-per-negative sets (af, claude_short, qwen, para_same): centrality ~0.97-1.0, marked_v1 0.74-0.88.
- Shared-edit sets (share3, ab_ac): centrality ~0.22-0.35, marked_v1 0.31-0.44 (last or near last).
- para_negs gets centrality to 0.50, but v2 (0.72) wins by spotting the one GT-worded choice.
- `mix_order.py`: mixing a one-edit cell with a shared cell gives the strict LB order only at
  para_full p=0.57 (one 0.01 step); starter-last and centrality-first hold for p up to ~0.45-0.65,
  but there v2 always leads the middle group by 0.05-0.10 and starter sits at 0.47-0.50, not 0.35.
