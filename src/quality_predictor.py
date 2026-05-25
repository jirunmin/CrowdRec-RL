"""
Worker Quality 预测模块。
用有标签 worker (quality_raw > 0) 的行为特征训练回归模型，
预测无标签 worker 的 quality。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 用于预测 quality 的特征列
# ============================================================
PREDICTOR_FEATURE_COLS = [
    "worker_total_entries",
    "worker_total_projects",
    "worker_win_count",
    "worker_win_rate",
    "worker_finalist_count",
    "worker_avg_score",
    "worker_category_entropy",
    "worker_active_days",
    "worker_pref_category",
    "worker_pref_sub_category",
    "worker_pref_industry",
]


def train_quality_predictor(worker_features_df: pd.DataFrame,
                             verbose: bool = True) -> dict:
    """
    用有标签 worker 训练 quality 预测模型。

    参数:
        worker_features_df: 来自 build_worker_features() 的输出
    返回:
        {"model": trained_model, "scaler": scaler, "cv_score": float}
    """
    wf = worker_features_df.copy()

    # 分离有标签和无标签的 worker
    labeled = wf[wf["worker_quality_raw"] > 0].copy()
    unlabeled = wf[wf["worker_quality_raw"] <= 0].copy()

    if verbose:
        print("\n" + "=" * 60)
        print("Step 3.5: Worker Quality 预测")
        print("=" * 60)
        print(f"  有标签 worker:   {len(labeled)}")
        print(f"  无标签 worker:   {len(unlabeled)}")

    if len(labeled) < 50:
        print("  Warning: labeled workers < 50, skipping prediction")
        return None

    # 准备训练数据
    X = labeled[PREDICTOR_FEATURE_COLS].fillna(0).values
    y = labeled["worker_quality_raw"].values  # 原始 0-100 分

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 GBDT 回归器
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # 交叉验证评估
    cv_scores = cross_val_score(model, X_scaled, y, cv=5,
                                 scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    if verbose:
        print(f"  CV MAE (5-fold): {cv_mae:.2f}  (score range 0-100)")
        print(f"  CV R² (5-fold):  {cross_val_score(model, X_scaled, y, cv=5, scoring='r2').mean():.3f}")

        # 特征重要性
        importances = model.feature_importances_
        print(f"\n  Feature importances:")
        for feat, imp in sorted(zip(PREDICTOR_FEATURE_COLS, importances),
                                  key=lambda x: -x[1]):
            print(f"    {feat:35s} {imp:.4f}")

    return {"model": model, "scaler": scaler, "cv_mae": cv_mae}


def predict_quality(predictor: dict,
                     worker_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    用训练好的模型预测所有 worker 的 quality。
    有标签的保留原值，无标签的用预测值。
    新增列: worker_quality_pred（0-1 归一化）。

    返回增强后的 worker_features_df。
    """
    if predictor is None:
        # 未训练，直接返回（用 median 填充作为预测值）
        wf = worker_features_df.copy()
        wf["worker_quality_pred"] = wf["worker_quality"]  # median fill
        return wf

    model = predictor["model"]
    scaler = predictor["scaler"]
    wf = worker_features_df.copy()

    # 预测所有 worker
    X_all = wf[PREDICTOR_FEATURE_COLS].fillna(0).values
    X_all_scaled = scaler.transform(X_all)
    y_pred_raw = model.predict(X_all_scaled)

    # Clip to [0, 100]
    y_pred_raw = np.clip(y_pred_raw, 0, 100)

    # 对于有标签的 worker，保留原始值；对于无标签的，用预测值
    wf["worker_quality_pred_raw"] = np.where(
        wf["worker_quality_raw"] > 0,
        wf["worker_quality_raw"],
        y_pred_raw
    )

    # 归一化到 [0, 1]
    wf["worker_quality_pred"] = wf["worker_quality_pred_raw"] / 100.0
    wf["worker_quality_pred"] = wf["worker_quality_pred"].clip(0.0, 1.0)

    print(f"  worker_quality_pred 统计: "
          f"mean={wf['worker_quality_pred'].mean():.3f}, "
          f"std={wf['worker_quality_pred'].std():.3f}, "
          f"median={wf['worker_quality_pred'].median():.3f}")

    return wf
