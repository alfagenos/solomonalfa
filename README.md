# Solomon MVP

Minimal working prototype of the financial AI architecture.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Free hosting

A simple option is Streamlit Community Cloud.

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Create a new Streamlit app from that repository.
4. Select `app.py` as the main file.
5. Deploy.

No Binance API key is required because the prototype uses public market-data endpoints.

## Architecture

Market data
→ feature engineering
→ ML classifier
→ probability distribution
→ signal
→ backtest
→ dashboard

## Important

This is a research prototype, not a profitable trading system by default.
Do not connect it to an exchange for live trading until the validation,
data quality, execution model, leakage checks and risk controls are substantially improved.
