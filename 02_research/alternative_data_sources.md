1. Alpha Vantage — NEWS_SENTIMENT endpoint (Best overall)

This is the cleanest drop-in for your dataframe. It returns a structured ticker_sentiment_score (float, -1 to +1) and ticker_sentiment_label per ticker per time window, straight from the API with no NLP on your end.

Key endpoint:

GET [https://www.alphavantage.co/query?function=NEWS_SENTIMENT](https://www.alphavantage.co/query?function=NEWS_SENTIMENT)
    &tickers=AAPL,MSFT,NVDA
    &time_from=20240101T0000
    &time_to=20240102T0000
    &apikey=YOUR_KEY
Free tier: 25 requests/day (standard free key) — you can batch multiple tickers per call (comma-separated), so 150 stocks can often fit in 5-10 calls
Historical: Yes, time_from/time_to supports going back several years
Output: Per-article sentiment + per-ticker aggregate score — trivially joins to your long df on (ticker, date)
Key constraint: The free key rate limit is real; you need to cache results to parquet and not re-fetch (you already have data/cache/)
Get a key at alphavantage.co — instant, no credit card.

1. StockTwits — Social Sentiment (Zero friction, no key needed)

Free, no API key required for public symbol streams. Returns bullish_count, bearish_count, and a sentiment label from the message stream.

GET [https://api.stocktwits.com/api/2/streams/symbol/AAPL.json](https://api.stocktwits.com/api/2/streams/symbol/AAPL.json)
Free tier: Truly free for public endpoints, no auth
Historical: No — only current/recent messages
Best for: Live deployment signal (run nightly, store the score in your cache)
Fit: Good for a cross-sectional rank signal (bullish ratio across your 150 names each day)
3. Finnhub — Company News + local FinBERT/VADER
Free tier gives you 60 API calls/minute, and their /company-news endpoint returns headlines and summaries per ticker per date range. You then score them locally.

Free tier: Very generous — 60 req/min, no monthly cap
Historical: Yes, several years back
Sentiment: Not built-in — you run vaderSentiment or transformers (FinBERT) locally on the headlines, which makes it fully reproducible and self-contained
Key advantage: VADER is deterministic (same text → same score forever), making it a stable, deployable signal
Key cost: ~5 min of local NLP per batch run, but parallelizable
import finnhub
client = finnhub.Client(api_key="YOUR_FREE_KEY")
news = client.company_news("AAPL", _from="2024-01-01", to="2024-01-07")
Get a key at finnhub.io — instant.

1. GDELT — For Historical Backtesting from 2020 (No key, truly free)

If you need sentiment going back to 2020 for your 150 tickers, GDELT is the only free option with that depth. It ingests global news and scores article tone (-100 to +100) in the GKG (Global Knowledge Graph).

Free tier: Completely free, no key, queryable via BigQuery (free tier) or their REST API
Historical: Goes back to 2015+
Fit: Best for building the historical training dataset; needs more wrangling (entity matching ticker → company name) but is very powerful
Reproducible: Yes — it's a static dataset
The simplest access pattern is the GDELT 2.0 Mentions table via BigQuery — BigQuery free tier gives you 1 TB/month of queries.

Recommended Approach for Your Framework
Given your setup (long-format df, data/cache/, look-ahead bias rules), the cleanest path is:

For deployment (live signal): Alpha Vantage NEWS_SENTIMENT — fetch nightly, store to parquet in data/cache/, merge into your feature store with merge_asof(..., direction='backward') so you never leak future sentiment into past bars.

For backtesting from 2020: Finnhub news + VADER locally — it's deterministic, free, and you own the scoring logic. One batch fetch stores everything to parquet once, then it's just a join.

The feature_store.py file you already have is the natural home for a build_sentiment_features(df, tickers, date_col) function that pulls from your cached parquet and aligns with merge_asof.

---



## **1. Can GDELT be used for live/daily deployment?**

**Yes — GDELT updates every 15 minutes.** It's not just a historical archive. The GDELT 2.0 GKG publishes a new batch file at:

[http://data.gdeltproject.org/gdeltv2/lastupdate.txt](http://data.gdeltproject.org/gdeltv2/lastupdate.txt)

For a daily strategy you'd fetch at market close (or a fixed time after), aggregate the last 24 hours of tone scores by company name, and that becomes your daily sentiment feature. The delay is ~15 minutes, which is completely fine for a daily bar strategy.

The real challenges for live GDELT use are not timeliness — they are:

- **Entity resolution**: GDELT doesn't index by ticker. It has `"Apple Inc"`, `"AAPL.US"` style strings in its `V2Organizations` column. You need a reliable ticker → company-name mapping table, and it occasionally misfires on ambiguous names.
- **Data volume**: Each 15-min chunk is a sizeable CSV. For 150 names you need to filter efficiently, or you're downloading GBs per day.

But it is genuinely deployable on a daily cadence.

---



## **2. Train on GDELT, deploy on Alpha Vantage — will the GBM degrade?**

**Yes, and this is a serious problem.** What you're describing is a classic **covariate shift / feature distribution mismatch**, and GBMs are more sensitive to it than most people realise.

Normalising both to [-1, +1] does **not** actually solve it.

**Bottom line:** Use GDELT end-to-end if you can solve the entity resolution problem — train on it, deploy on it, and you have zero distribution shift. If you do mix sources, the cross-sectional rank transform is the minimal change that makes the feature meaningful across both, without touching the rest of your training pipeline.



note: finnhub is not suitable as does not reach 2020 (only 1 year of historical data)