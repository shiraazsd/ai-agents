# Working in this repo

- Skills live in `common/skills/` or `teams/<team>/skills/`. Scaffold with `mkt new-skill <plugin> <skill>`.
- Keep `SKILL.md` short: the description says *when* to use it; the body says *how*. Detail goes in `references/`.
- Tools are typed, documented top-level functions in `scripts/tools.py` ending with `if __name__ == "__main__": run()`.
  Use `skillkit.session()` for HTTP; never read credentials directly.
- Every skill needs `evals.yaml`; add `fixtures/` for any HTTP the tools make.
- Before finishing: `mkt sync && mkt validate && pytest -q`.
