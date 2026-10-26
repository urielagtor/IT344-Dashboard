import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

customers = pd.read_csv(os.path.join(DATA, "customers.csv"))
merchants = pd.read_csv(os.path.join(DATA, "merchants.csv"))
train = pd.read_csv(os.path.join(DATA, "transactions_train.csv"))
evalx = pd.read_csv(os.path.join(DATA, "transactions_eval.csv"))
labels = pd.read_csv(os.path.join(DATA, "evaluation_labels.csv"))

# ── Clean ──────────────────────────────────────────────────────────────────
customers = customers.drop_duplicates(subset="customer_id")
merchants = merchants.drop_duplicates(subset="merchant_id")
train = train.drop_duplicates(subset="transaction_id")
evalx = evalx.drop_duplicates(subset="transaction_id")


def clean_str(s):
    return str(s).strip().lower() if pd.notna(s) else s


for col in ["is_online", "card_present"]:
    train[col] = train[col].apply(clean_str)
    evalx[col] = evalx[col].apply(clean_str)
merchants["merchant_category"] = merchants["merchant_category"].apply(clean_str)
merchants["is_international"] = merchants["is_international"].apply(clean_str)

print(f"Train rows after cleaning: {len(train):,}   Eval rows: {len(evalx):,}")


# ── Join ───────────────────────────────────────────────────────────────────
def enrich(txn):
    txn = txn.merge(customers[["customer_id", "account_age_days"]], on="customer_id", how="left")
    txn = txn.merge(merchants[["merchant_id", "merchant_category", "is_international"]],
                    on="merchant_id", how="left")
    return txn


train, evalx = enrich(train), enrich(evalx)

# ── Features ───────────────────────────────────────────────────────────────
cat_cols = ["is_online", "card_present", "merchant_category", "is_international"]
for col in cat_cols:
    train[col] = train[col].fillna("unknown")
    evalx[col] = evalx[col].fillna("unknown")
    le = LabelEncoder().fit(pd.concat([train[col], evalx[col]]).unique())
    train[col + "_enc"] = le.transform(train[col])
    evalx[col + "_enc"] = le.transform(evalx[col])

features = [
    "transaction_amount", "distance_from_home_miles", "hours_since_last_txn",
    "ratio_to_median_purchase", "num_txns_last_24h", "account_age_days",
    "is_online_enc", "card_present_enc", "merchant_category_enc", "is_international_enc",
]

# ── Model ──────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                               random_state=42, n_jobs=-1)
model.fit(train[features].fillna(0), train["is_fraud"])
evalx["fraud_probability"] = model.predict_proba(evalx[features].fillna(0))[:, 1]

# ── Attach actual outcomes ─────────────────────────────────────────────────
out = evalx.merge(labels, on="transaction_id", how="inner")
y, p = out["is_fraud"].values, out["fraud_probability"].values
prec, rec, _ = precision_recall_curve(y, p)
print(f"AUC-PR: {auc(rec, prec):.3f}")
for cut in (0.5, 0.2):
    yp = (p >= cut).astype(int)
    print(f"cutoff {cut}: precision {precision_score(y, yp):.3f}  "
          f"recall {recall_score(y, yp):.3f}  F1 {f1_score(y, yp):.3f}")


keep = ["transaction_id", "transaction_datetime", "transaction_amount", "merchant_category",
        "is_online", "card_present", "is_international", "distance_from_home_miles",
        "num_txns_last_24h", "fraud_probability", "is_fraud"]
out[keep].to_csv(os.path.join(DATA, "predictions.csv"), index=False)
print(f"Wrote predictions.csv ({len(out):,} rows)")
