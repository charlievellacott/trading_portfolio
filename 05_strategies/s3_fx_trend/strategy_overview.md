# TSMOM strategy overview

## The starting point
1. Measure whether that pair has been trending up or down over some lookback.
2. Go long if up, short if down (or flat if weak).
3. Size by risk (usually inverse vol).
4. Combine pairs into a book; often scale the whole book to a portfolio vol target.
5. Pay realistic FX costs (your OANDA model).

- This is the starting point. Everything else (breakout vs MA, bands, carry gate) is a variant of how you measure “trending” or how you risk-manage a weak but persistent premium.

## Additions
- Look at momentum as a recent breakout occurs
- Look at correlation (bet of weakly correlated trends)
- Weight by conviction (TSMOM + breakout significance), then per asset vol, then portfolio level vol
- Blend horizons (1/3/12M)
- Macro gate (confim that the trend aligns with the macros)
Consider:
> Carry (rate differential) as the underlying signal --> trend aligns (multi-period) --> breakout 
> Value (PPP / real rate) as the underlying signal --> trend aligns (multi-period) --> breakout 
> Event Pricing (forecast/expected vs actual = surprise score: good or bad for pair? conidering currency order) -->  trend aligns OR breakout (period needs to be further considered)

## How will the time frame and trading sessions effect implementation?
- 

## Why it happens?
- Underreaction / Slow reaction to economic news
> eg) Rates, inflation, growth news hit FX gradually.
- Stop losses amplify moves once a level breaks (leading to continuation)
- Trends attract followers (and in FX this persists for longer than in individual stocks)
- Crisis convexity 

## Expectations
- This is a risk premimum not an inefficiency. So expect modest results eg)
> Sharpe ~ 0.3-0.8
- TSMOM != CSMOM --> TSMOM is better as it does not depend on the entire universe performing well, only on that stock compared to its past. It stops trades being taken just for the sake of trading the universe.