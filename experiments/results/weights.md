(crowdrec) zkz@localhost:~/course/RL/CrowdRec-RL$ python baselines/find_weights_max_reward.py
============================================================
最大化 Reward — 线性回归权重学习
============================================================

训练事件: 975195 (正:335513, 负:639682)

============================================================
Greedy-Worker 特征 → 预测 Worker Reward
============================================================

  目标: reward_worker
  R²: 0.0856
  预测值范围: [0.00, 4.50], mean=0.50

                    | 特征          | 权重    | 影响力  |
                    | ------------- | ------- | ------- |
                    | awards        | 0.0001  | 0.0732  |
                    | total_match   | 0.2024  | 0.2002  |
                    | quality_match | -0.1914 | -0.0520 |
                    | featured      | 0.0995  | 0.0489  |
                    | avg_score     | 0.0488  | 0.0709  |

============================================================
Greedy-Worker 特征 → 预测 Requester Reward
============================================================

  目标: reward_requester
  R²: 0.0886
  预测值范围: [0.00, 3.00], mean=0.58

                    | 特征          | 权重    | 影响力  |
                    | ------------- | ------- | ------- |
                    | awards        | 0.0001  | 0.0806  |
                    | total_match   | 0.2307  | 0.2282  |
                    | quality_match | -0.6297 | -0.1712 |
                    | featured      | 0.1194  | 0.0587  |
                    | avg_score     | 0.1355  | 0.1969  |

============================================================
Greedy-Requester 特征 → 预测 Worker Reward
============================================================

  目标: reward_worker
  R²: 0.1018
  预测值范围: [0.00, 4.50], mean=0.50

                    | 特征           | 权重   | 影响力 |
                    | -------------- | ------ | ------ |
                    | avg_score      | 0.0288 | 0.0419 |
                    | has_winner     | 0.1022 | 0.0374 |
                    | worker_count   | 0.0041 | 0.1473 |
                    | match_industry | 0.1889 | 0.0859 |
                    | entry_count    | 0.0004 | 0.0563 |

============================================================
Greedy-Requester 特征 → 预测 Requester Reward
============================================================

  目标: reward_requester
  R²: 0.1009
  预测值范围: [0.00, 3.00], mean=0.58

                    | 特征           | 权重   | 影响力 |
                    | -------------- | ------ | ------ |
                    | avg_score      | 0.0383 | 0.0556 |
                    | has_winner     | 0.1145 | 0.0419 |
                    | worker_count   | 0.0048 | 0.1711 |
                    | match_industry | 0.2006 | 0.0912 |
                    | entry_count    | 0.0004 | 0.0629 |

============================================================
可直接复制到代码的参数
============================================================

--- Greedy-Worker（worker reward）→ greedy_worker.py ---
  w_awards: float = 0.00011736,
  w_match: float = 0.20240058,
  w_quality_gap: float = 0.19140123,
  w_featured: float = 0.09949123,
  w_score: float = 0.04879118,

--- Greedy-Worker（requester reward）→ greedy_worker.py ---
  w_awards: float = 0.00012926,
  w_match: float = 0.23073093,
  w_quality_gap: float = 0.62967077,
  w_featured: float = 0.11942735,
  w_score: float = 0.13545993,

--- Greedy-Requester（worker reward）→ greedy_requester.py ---
  w_quality: float = 0.02883347,
  w_winner: float = 0.10221480,
  w_worker_count: float = 0.00414921,
  w_match: float = 0.18886589,
  w_entry_count: float = 0.00036758,

--- Greedy-Requester（requester reward）→ greedy_requester.py ---
  w_quality: float = 0.03825913,
  w_winner: float = 0.11452780,
  w_worker_count: float = 0.00482159,
  w_match: float = 0.20055764,
  w_entry_count: float = 0.00041078,

============================================================
全特征模型（合并 Worker + Requester 特征）
============================================================

--- 预测 Worker Reward ---

  目标: reward_worker
  R²: 0.1332
  预测值范围: [0.00, 4.50], mean=0.50

                    | 特征          | 权重    | 影响力  |
                    | ------------- | ------- | ------- |
                    | awards        | 0.0000  | 0.0212  |
                    | match_cat     | 0.2202  | 0.0869  |
                    | match_sub     | 0.1178  | 0.0526  |
                    | match_ind     | 0.1381  | 0.0628  |
                    | quality_match | -0.2665 | -0.0724 |
                    | featured      | 0.0476  | 0.0234  |
                    | avg_score     | 0.0719  | 0.1046  |
                    | has_winner    | 0.0942  | 0.0345  |
                    | worker_count  | 0.0023  | 0.0815  |
                    | entry_count   | 0.0005  | 0.0824  |

--- 预测 Requester Reward ---

  目标: reward_requester
  R²: 0.1387
  预测值范围: [0.00, 3.00], mean=0.58

                    | 特征          | 权重    | 影响力  |
                    | ------------- | ------- | ------- |
                    | awards        | 0.0000  | 0.0197  |
                    | match_cat     | 0.2644  | 0.1043  |
                    | match_sub     | 0.1263  | 0.0563  |
                    | match_ind     | 0.1482  | 0.0674  |
                    | quality_match | -0.7128 | -0.1938 |
                    | featured      | 0.0588  | 0.0289  |
                    | avg_score     | 0.1611  | 0.2342  |
                    | has_winner    | 0.1042  | 0.0382  |
                    | worker_count  | 0.0026  | 0.0917  |
                    | entry_count   | 0.0007  | 0.0997  |

--- 全特征（worker reward）→ 统一策略 ---
  w_awards: float = 0.00003400,
  w_match_cat: float = 0.22017864,
  w_match_sub: float = 0.11781708,
  w_match_ind: float = 0.13808125,
  w_quality_match: float = -0.26647046,
  w_featured: float = 0.04761916,
  w_avg_score: float = 0.07194080,
  w_has_winner: float = 0.09423370,
  w_worker_count: float = 0.00229748,
  w_entry_count: float = 0.00053837,

--- 全特征（requester reward）→ 统一策略 ---
  w_awards: float = 0.00003164,
  w_match_cat: float = 0.26436990,
  w_match_sub: float = 0.12628057,
  w_match_ind: float = 0.14821165,
  w_quality_match: float = -0.71283859,
  w_featured: float = 0.05883334,
  w_avg_score: float = 0.16107203,
  w_has_winner: float = 0.10417320,
  w_worker_count: float = 0.00258450,
  w_entry_count: float = 0.00065133,