# Contributing

Thanks for your interest in contributing to MESSAI's open-source packages.

> **This repository is a downstream mirror.** Source of truth lives in the
> private `messai-ai` monorepo; this mirror is updated automatically on each
> release. PRs against this mirror will be auto-closed by a bot with a friendly
> redirect to this guide.

## How to contribute

### 🐛 Reporting bugs / 🙋 asking questions

Open a [GitHub Issue](../../issues) or start a [Discussion](../../discussions)
here on the mirror repo. Maintainers monitor both. Bug reports should include:

- The package version affected (from `package.json` or `pyproject.toml`)
- A minimal reproduction
- What you expected vs. what happened

### 💡 Proposing changes

Because this repo is a mirror, we can't accept PRs directly. To propose a
change:

1. **Open an Issue first** describing the change — what problem it solves, what
   the API/data shape would look like. This is the fastest path to alignment
   before code is written.
2. **If you'd like to contribute the code yourself**: write your patch as a fork
   of this mirror, push it to a branch, and reference it in your Issue. A
   maintainer will cherry-pick it into the upstream monorepo on your behalf and
   credit you via `Co-authored-by:`.

### 📊 Data corrections (data packages only)

For the data packages (`@messai-io/mess-parameters`,
`@messai-io/mess-materials`, `@messai-io/mess-datasets-catalog`,
`@messai-io/mess-learning`), data corrections are especially welcome. Open an
Issue with:

- The dataset row(s) affected (slug or DOI)
- The correction + a citation source
- Whether this is a typo, a methodology refinement, or new evidence

## Code of conduct

By participating in this project you agree to abide by the
[Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).

## Citation

If you use any of these packages in research, please cite per the `CITATION.cff`
in the relevant package. Each tagged release also mints a Zenodo DOI.

## License

See [LICENSE](./LICENSE). Data packages are CC BY 4.0; code packages are MIT.
