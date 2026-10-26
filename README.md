# Fraud Model Check-in (Streamlit prototype)

    pip install -r requirements.txt
    streamlit run app.py

`data/predictions.csv` is already included (150,000 test transactions with the
model's fraud score next to the actual outcome), so the app runs right away.

To rebuild it from the raw files, put customers.csv, merchants.csv,
transactions_train.csv, transactions_eval.csv and evaluation_labels.csv in `data/`
and run `python build_predictions.py` (~40 s).

To show your exact Week 3 scores instead, save your submission file as
`data/submission.csv` (columns: transaction_id, fraud_probability). The app
picks it up automatically and the notes panel says which scores are in use.
