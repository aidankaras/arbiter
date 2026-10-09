# Running the pipeline

The commands in the README's quickstart, with what each one does and why it
behaves as it does.

## Ingesting a day

```bash
uv run arbiter ingest 2026-08-10
```

```
insider: 755
redflag: 46
rejected: 0
unpriceable: 34
```

The command fetches every Form 4 and 8-K accepted on the given day, extracts the
qualifying events, and writes them to `data/events/{domain}/{date}.parquet`.
Re-running a date replaces that date's partition, so a failed run is repeated
rather than repaired.

The last two counts are different facts and are recorded separately. `rejected`
is filings that could not be parsed, which is a defect to investigate; a day
where too many fail is refused outright rather than recorded, on the reasoning
that a format change has broken it. `unpriceable` is filings parsed correctly
whose issuer has no listed common stock — insiders at companies with only
registered debt file Form 4 like anyone else — which is an ordinary property of
the population and is excluded from that share.

EDGAR ingestion needs no credentials beyond the contact string the SEC requires
(`SEC_USER_AGENT` in `.env`).

## Labelling a day

```bash
uv run arbiter resolve 2026-08-10 insider
uv run arbiter resolve 2026-08-10 redflag
```

```
labels-insider: 751
labels-redflag: 40
```

Labelling lags ingestion. The horizon is five sessions for insider events and
twenty for red flags, and the consolidated tape will not serve a window ending
on the current session, so a day becomes measurable only after its window has
closed. A day holding any event whose window is still open is not labelled at
all: `arbiter resolve` writes nothing and names the session after which to
rerun it, and `backfill` reports the day as pending rather than complete, so a
later run picks it up. A partial day would otherwise look finished and its open
events would never be labelled. Issuers that cannot
be priced are recorded under `unpriceable/` beside the labels, so a thin day
stays distinguishable from a day whose filers were unlistable.

Market data requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.

## Building a history

```bash
uv run arbiter backfill 2026-03-02 2026-08-14 --every 4
```

`backfill` ingests and labels a range of trading days, writing its progress as
it goes. Days already stored are skipped, so an interrupted run is restarted
rather than repaired.

`--every` is the lever on how long a run takes. A day costs the same regardless
of how many of its filings qualify, because each filing must be fetched to find
out and the acceptance timestamp that makes an event point-in-time is not
carried in EDGAR's bulk index. Volume varies roughly fourfold across the year —
736 Form 4 filings on one sampled day against 3,001 in early March, when annual
grants and vesting cluster — so a day takes between three and fourteen minutes.
Sampling every Nth trading day spreads observations across months at the cost of
a contiguous block, which matters because events filed on one day share a market
factor the sector benchmark only partly removes.

## Measuring

```bash
uv run arbiter report --domain insider
```

`report` fits the baseline arm on the earlier fraction of the stored days and
scores it on the rest, writing `reports/baseline-insider.md` and the figure
beside it. The split is chronological, never random: a random split would place
events from one day on both sides and report as skill what is partly memory.
Days whose outcomes were still open when the last fitted day's forecasts were
made are withheld from both sides. [`methodology.md`](methodology.md) covers
the evaluation design in full.
