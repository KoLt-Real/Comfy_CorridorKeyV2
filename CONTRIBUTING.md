# Contributing

Bug reports and patches are welcome. Two things about this repository are unusual,
and both are easy to break without noticing — please read them first.

## 1. This project is non-commercial, and that is not negotiable

It is a derivative work under **CC BY-NC-SA 4.0 plus Corridor Digital's additional
terms** (see `LICENSE` and `NOTICE`). By opening a pull request you agree that your
contribution is released under those same terms, ShareAlike included.

Practical consequences:

- Do **not** add a permissive SPDX header, a `license = "Apache-2.0"` line, or any
  other notice that would misstate the terms. GitHub shows this repository as
  `NOASSERTION` because the licence is non-standard — that is correct, please leave it.
- Do **not** remove or trim `NOTICE`. Attribution to "CorridorKey" is an obligation,
  not a courtesy.
- Do **not** contribute code lifted from a permissively licensed project without
  saying so in `NOTICE`, with the upstream commit pinned.

## 2. `pipeline/vendor/` is vendored at a pinned commit

It is upstream code, listed with its commit hash in `NOTICE`. **Do not refactor,
reformat, rename or "improve" anything in there** — a diff against upstream is what
lets us pull a fix later. If behaviour has to change, wrap it from `pipeline/`.

The other invariants (lazy `torch` imports, the `min_island_size` floor that
silently blackens a whole matte if it is bypassed, and the rest) live in `AGENTS.md`.
Read that file before your first patch; it is short and every entry is there because
the thing it describes has already gone wrong once.

## Running the tests

```bash
python3 tests/test_pipeline_guards.py
```

They are deliberately torch-free, so they run anywhere — that is a direct consequence
of the lazy-import invariant, and it is worth preserving. Full-pipeline verification
needs a GPU and is not part of the suite: use `example_workflows/` for that, and say
in your PR which GPU and how much VRAM you ran it on.

## Style

Everything is in **English** — code, comments, docstrings, test names, log messages,
commit messages. Match the surrounding code rather than the style you would pick.
