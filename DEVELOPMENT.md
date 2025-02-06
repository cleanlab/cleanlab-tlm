# Development

This project uses the [Hatch] project manager ([installation instructions][hatch-install]).

Hatch automatically manages dependencies and runs testing, type checking, and other operations in isolated [environments][hatch-environments].

[Hatch]: https://hatch.pypa.io/
[hatch-install]: https://hatch.pypa.io/latest/install/
[hatch-environments]: https://hatch.pypa.io/latest/environment/

## Testing

You can run the tests on your local machine with:

```bash
hatch test
```

The [`test` command][hatch-test] supports options such as `-c` for measuring test coverage, `-a` for testing with a matrix of Python versions, and appending an argument like `tests/test_prompt.py::test_single_prompt` for running a single test.

[hatch-test]: https://hatch.pypa.io/latest/tutorials/testing/overview/

## Type checking

You can run the [mypy static type checker][mypy] with:

```bash
hatch run types:check
```

[mypy]: https://mypy-lang.org/

## Formatting and linting

You can run the [Ruff][ruff] formatter and linter with:

```bash
hatch fmt
```

This will automatically make [safe fixes][fix-safety] to your code. If you want to only check your files without making modifications, run `hatch fmt --check`.

[ruff]: https://github.com/astral-sh/ruff
[fix-safety]: https://docs.astral.sh/ruff/linter/#fix-safety

## Pre-commit

You can install the pre-commit hooks to automatically run type checking, formatting, and linting on every commit.

First, install [pre-commit][pre-commit], for example, with [pipx]:

```bash
pipx install pre-commit
```

Then, install the hooks:

```bash
pre-commit install
```

[pipx]: https://pipx.pypa.io/

## Packaging

You can use [`hatch build`][hatch-build] to create build artifacts, a [source distribution ("sdist")][sdist] and a [built distribution ("wheel")][bdist].

You can use [`hatch publish`][hatch-publish] if you want to manually publish build artifacts to [PyPI][pypi].

[hatch-build]: https://hatch.pypa.io/latest/build/
[sdist]: https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist
[bdist]: https://packaging.python.org/en/latest/glossary/#term-Built-Distribution
[hatch-publish]: https://hatch.pypa.io/latest/publish/
[pypi]: https://pypi.org/

### Automated releases

Automated releases are handled by the [release workflow][release-workflow] which is triggered by pushing a new tag to the repository. To create a new release:

1. Bump the version in `src/cleanlab_tlm/__about__.py`. You can use the [`hatch version`][hatch-version] command to do this.
2. Ensure that the release notes are updated in [`CHANGELOG.md`][changelog]:
   - You should update the `[Unreleased]` header to the new version and add a new `[Unreleased]` section at the top of the file.
   - You should update the link for the `[Unreleased]` code and add a new link to the code diff for the new version.
3. Create a PR and merge these changes into the `main` branch.
4. After the PR is merged into `main`, create a new release tag by running `git tag v<output of hatch version>` (i.e. `git tag v0.0.1`).
5. Push the tag to the repository by running `git push origin <tag>`.
6. This will trigger the release workflow which will build the package, create a release on GitHub, and publish the package version to PyPI. The GitHub release notes will be automatically generated from the [changelog].

[release-workflow]: .github/workflows/release.yml
[hatch-version]: https://hatch.pypa.io/latest/version/#updating
[changelog]: CHANGELOG.md

## Continuous integration

Testing, type checking, and formatting/linting is [checked in CI][ci].

[ci]: .github/workflows/ci.yml
