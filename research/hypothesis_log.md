# Hypothesis Log


| ID    | Date       | Asset    | Factor           | Status  |
| ----- | ---------- | -------- | ---------------- | ------- |
| H-001 | 2026-02-07 | Equities | SMART Beta       | PENDING |
| H-002 | 2026-02-07 | Equities | Size & Value     | PENDING |
| H-003 | 2026-02-07 | Equities | Sentiment        | PENDING |
| H-004 | 2026-02-07 | Equities | HMM Trend Regime | PENDING |
| H-005 | 2026-02-07 | Equities | Autocorrelation  | PENDING |


---

## H-001 · Equities · SMART Beta · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                             |
| **Hypothesis**         | The key weakness for a stock picking strategy is when the market is down. My having lower beta stocks the amount of alpha would be increased.                                                                                                                       |
| **Economic rationale** | When a crash occurs the predicted (GBM) top n stocks (if have low alpha) will still move higher irrespectable of the market.                                                                                                                                        |
| **Data required**      | —                                                                                                                                                                                                                                                                   |
| **Test to complete**   | What is the best way of calculating beta?                                                                                                                                                                                                                           |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                   |
| **Notes**              | SMART beta is different to BETA as decisions are not made entirely based off beta (instead other factors but infulenced by beta). Do I want smart beta or instead just low-beta / low-volatility anomaly as a factor? (Frazzini & Pedersen, "Betting Against Beta") |


---



## H-002 · Equities · Size & Value · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                                                      |
| **Hypothesis**         | 1. Use a "normalized rate of change of valuation (of a stock)"? (size momentum or a market-cap growth factor). 2. Volume spikes during trending markets could indicate a reversal and vice versa for a ranging market.                                                                       |
| **Economic rationale** | Volume spikes just after (or while still in) a highly bullish market indicate high selling pressure. RoC in valuation dictacts the percieved growth of a company.                                                                                                                            |
| **Data required**      | —                                                                                                                                                                                                                                                                                            |
| **Test to complete**   | Explore how change in size is effected based on the direction of the stock before. Look at the effects of volume spikes when the market is moving in different directions - compare the volume spike senarios to controls (where there are no volume spikes) but similar movements in price. |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                                                            |
| **Notes**              | What actually is size?                                                                                                                                                                                                                                                                       |


---



## H-003 · Equities · Sentiment · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                                 |
| **Hypothesis**         | Higher sentiment = higher movement in price, etc.                                                                                                                                                                                                       |
| **Economic rationale** | Sentiment is used to explain the percieved value of a stock, thus depending on vibe certain retail investors may back off or go into a specific stock.                                                                                                  |
| **Data required**      | —                                                                                                                                                                                                                                                       |
| **Test to complete**   | Firstly, is there a tradable signal? (if applicable: use the past week of news headlines to create a sentiment score using FinBert) Then, explore the decay of the signal (especially if using GDELT as need to decide on a window of articles to use). |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                       |
| **Notes**              | Need to work out how to get a hold of the data first.                                                                                                                                                                                                   |


---



## H-004 · Equities · HMM Trend Regime · 2026-02-07


| Field                  |                                                                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                             |
| **Hypothesis**         | Surely buying and selling should align with the stocks in a bull and bear market - this focuses on momentum as opposed to mean reversion.                                           |
| **Economic rationale** | Momentum                                                                                                                                                                            |
| **Data required**      | —                                                                                                                                                                                   |
| **Test to complete**   | Look at Hurst exponent or variance ratio tests. No matter where the model ranked stocks if the trade direction does not match the market direction (HMM state) then don't trade it. |
| **Alphalens summary**  | —                                                                                                                                                                                   |
| **Notes**              | Test idea with a backtest and a plot that details the agreement between the final direction, prediction and the HMM state.                                                          |


---



## H-005 · Equities · Autocorrelation · 2026-02-07


| Field                  |                                                                                                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**             | PENDING                                                                                                                                                                                                                                            |
| **Hypothesis**         | Multiplying (or the correct linear algebra operation) a prediction matrix (ranking) by a correlation matrix of the stock universe (over a given window) will tell you if the predictions align (or conflict) with each other via the correlations. |
| **Economic rationale** | Autocorrelation                                                                                                                                                                                                                                    |
| **Data required**      | —                                                                                                                                                                                                                                                  |
| **Test to complete**   | —                                                                                                                                                                                                                                                  |
| **Alphalens summary**  | —                                                                                                                                                                                                                                                  |
| **Notes**              | Test idea after creating predictions using the model, in a notebook or a backtest.                                                                                                                                                                 |


---



## [Asset Class] — [Factor Name] — [Date proposed]

Hypothesis: (one sentence — why should this predict returns?) 
Economic rationale: (2-3 sentences — what behaviour/structural effect causes this) 
Data required: 
Status: PENDING / KEPT / KILLED 
Test to complete: 
Alphalens summary: (filled in after testing) 
Notes: