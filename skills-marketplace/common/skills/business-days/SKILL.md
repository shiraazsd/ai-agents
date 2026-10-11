---
name: business-days
description: Count business days (Mon-Fri) between dates or find the date N business days away. Use for SLAs, due dates and "working days" questions.
---

# Business days

Never count days by hand — always use the tools:

- `business_days(start, end)` — Mon-Fri days in `[start, end)`.
- `add_business_days(start, days)` — date that is `days` business days after `start`.

Dates are ISO `YYYY-MM-DD`. Holidays are not excluded; say so if it matters.

From a shell: `python scripts/tools.py business_days --start 2026-01-05 --end 2026-01-19`
