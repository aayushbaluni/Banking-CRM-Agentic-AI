"""Train an XGBoost loan propensity model on the seeded CRM data."""
import sys
import os
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy.orm import Session
from app.db.database import SessionLocal, init_db
from app.db.schema import Customer
from app.config import settings


EDU_MAP = {
    "Undergraduate": 0,
    "Graduate": 1,
    "Post Graduate": 2,
    "Professional": 3,
    "Doctorate": 4,
}

# Features must match scoring_tools._extract_features order
FEATURE_NAMES = [
    "monthly_income", "account_balance", "credit_score", "age",
    "num_products", "has_salary_account", "has_fixed_deposit",
    "months_as_customer", "avg_monthly_txn_amount", "num_monthly_txns",
    "last_product_purchase_months", "education_encoded",
]


def extract_features(c: Customer) -> list[float]:
    return [
        c.monthly_income,
        c.account_balance,
        c.credit_score,
        c.age,
        c.num_products,
        int(c.has_salary_account),
        int(c.has_fixed_deposit),
        c.months_as_customer,
        c.avg_monthly_txn_amount,
        c.num_monthly_txns,
        c.last_product_purchase_months,
        EDU_MAP.get(c.education, 1),
    ]


def synthetic_label(c: Customer) -> int:
    """
    Derive a synthetic conversion label using banking domain heuristics.
    Targets ~12-15% positive rate to simulate realistic banking imbalance.
    Only the top-tier customers across ALL criteria get a positive label.
    """
    if c.has_personal_loan:
        return 0  # already has loan — not a conversion target

    score = 0
    # Credit score — high bar
    if c.credit_score >= 780:
        score += 4
    elif c.credit_score >= 750:
        score += 3
    elif c.credit_score >= 720:
        score += 1

    # Income — must be genuinely high
    if c.monthly_income > 120000:
        score += 4
    elif c.monthly_income > 80000:
        score += 2
    elif c.monthly_income > 50000:
        score += 1

    # Relationship depth
    if c.has_salary_account:
        score += 2
    if c.months_as_customer > 48:
        score += 2
    elif c.months_as_customer > 24:
        score += 1

    # Balance signals
    if c.account_balance > 2000000:
        score += 2
    elif c.account_balance > 800000:
        score += 1

    # Activity signal
    if c.num_monthly_txns > 20:
        score += 1

    # Noise for realism (small)
    score += np.random.randint(-1, 2)
    return int(score >= 10)  # high threshold → ~12-15% positive rate


def train():
    init_db()
    db: Session = SessionLocal()

    customers = db.query(Customer).all()
    db.close()

    if not customers:
        print("No customers in DB. Run seed.py first.")
        return

    np.random.seed(42)
    X = np.array([extract_features(c) for c in customers])
    y = np.array([synthetic_label(c) for c in customers])

    print(f"Dataset: {len(X)} samples, {y.sum()} positives ({y.mean():.1%} conversion rate)")

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from imblearn.over_sampling import SMOTE

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    print("\nTop feature importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")

    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    joblib.dump(model, settings.model_path)
    print(f"\nModel saved to {settings.model_path}")


if __name__ == "__main__":
    train()
