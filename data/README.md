# Data root

This directory is the canonical data layout.  Code lives in `dataset/`;
collected episode tensors live here.

```
data/
  specs/                     # fixed scenario definitions (tracked)
    source/
      train_10000.json
      val_1000.json
    train_2000.json          # fixed train scene set, seeds 1000..2999
    val_500.json             # fixed held-out validation scenes, seeds 5000000..5000499
    test_500.json            # fixed final test scenes, seeds 5000500..5000999
    mini_train_100.json      # smoke-test subset of train_2000
    mini_val_20.json         # smoke-test subset of val_500
  splits/                    # frozen seed manifests (tracked)
    train_2000.json
    val_500.json
    test_500.json
    mini_train_100.json
    mini_val_20.json
  episodes/                  # collected episode tensors (gitignored; large)
    train_2000/
      il_expert/             # IL/BC expert data, 2000 scenes
      il_dagger/             # DAgger student data, same 2000 scenes
    val_500/
      expert_reference/      # expert labels for offline BC validation only
    mini/
      il_train_100/          # 100-scene IL smoke subset
      dagger_train_100/      # 100-scene DAgger smoke subset
      val_20_expert/         # 20-scene validation smoke subset
  manifests/
    datasets.json            # registry: purpose, size, source
```

## Fixed scene sets

| name | scenes | seeds | role |
|---|---:|---|---|
| train_2000 | 2000 | 1000..2999 | training for IL / DAgger; RL env seeds |
| val_500 | 500 | 5000000..5000499 | held-out model selection / closed-loop validation |
| test_500 | 500 | 5000500..5000999 | final test only; do not tune |
| mini_train_100 | 100 | 1000..1099 | smoke tests |
| mini_val_20 | 20 | 5000000..5000019 | smoke tests |

`train_2000` and `val_500` are scene-disjoint.  `val_500` and `test_500`
are also disjoint.  Never train on validation or test scenes.

## Episode pools

`episodes/train_2000/il_expert` and `episodes/train_2000/il_dagger` cover the
same 2000 train scenes but are different policies/stages.  DAgger/RL may add
new episodes on the same train scenes; they must not change the split.

`episodes/val_500/expert_reference` is expert data on the held-out 500
validation scenes.  It is for offline validation/stability checks, not
training.

`val_500` is used for checkpoint/model selection.  `test_500` is final-only:
do not tune or select on it.

## Legacy

Superseded data and generated specs are under `archive/` and are not part of
the canonical path.  Do not use them for new runs.
