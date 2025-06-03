# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.5] - 2025-06-03

- Update link in docstring

## [1.1.4] - 2025-05-30

### Changed

- Update default model to `gpt-4.1-mini`

## [1.1.3] - 2025-05-13

### Changed

- Add server side max_timeout
- Add validation check for timeout to be > 0

## [1.1.2] - 2025-05-05

### Changed

- Update `query_ease` default criteria
- Add getter functions for `_TLM_DEFAULT_MODEL`, `_DEFAULT_TLM_QUALITY_PRESET`, `_TLM_DEFAULT_CONTEXT_LIMIT`, `_TLM_MAX_TOKEN_RANGE`.
- Add unit tests for the getter functions.

## [1.1.1] - 2025-04-23

### Changed

- Improved validation + error messages for TLM's `custom_eval_criteria`.
- Changed TLMOptions text in docs to have link to TLMOptions class

## [1.1.0] - 2025-04-21

### Changed

- All `.prompt()` / `.get_trustworthiness_score()` / `.generate()` / `.score()` methods will now catch any errors and return `null` values alongside a log of why the exception occurred
- `try_` methods are deprecated and will share the same functionality as the "non-try" methods

## [1.0.22] - 2025-04-18

### Added

- Update `response_helpfulness` default criteria

## [1.0.21] - 2025-04-17

### Added

- Add new OpenAI models: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3`, `o4-mini`

## [1.0.20] - 2025-04-15

### Added

- Add new options for `similarity_metric`: `code` and `discrepancy`

### Fixed

- Better validation / error message for invalid `evals` arguments in TrustworthyRAG

## [1.0.19] - 2025-04-14

### Added

- Add batch processing message

## [1.0.18] - 2025-04-11

### Added

- Add `score_async` method to TrustworthyRAG

## [1.0.16] - 2025-04-02

### Added

- TrustworthyRAG set retries

## [1.0.15] - 2025-04-01

### Added

- Update `context_sufficiency` default criteria

## [1.0.14] - 2025-03-31

### Added

- Add batch support to TrustworthyRAG

## [1.0.13] - 2025-03-20

### Added

- Add `response_groundedness` evaluation to TrustworthyRAG default evals
- Update `context_sufficiency` default prompt

## [1.0.10] - 2025-03-18

### Added

- Add `response_helpfulness` evaluation to TrustworthyRAG default evals

## [1.0.9] - 2025-03-18

### Added

- Add TrustworthyRAG

## [1.0.8] - 2025-03-06

### Added

- Custom evaluation supports multiple criteria
- Add support for `task` argument during TLM initialization

## [1.0.6] - 2025-03-03

### Added

- Add support for `gpt-4o-2024-11-20`

## [1.0.5] - 2025-02-27

### Added

- Add support for `gpt-4.5-preview`

## [1.0.4] - 2025-02-24

### Added

- Add `embedding` and `embedding_large` as new `similarity_measure` options
- Add support for Claude 3.7 Sonnet

## [1.0.3] - 2025-02-19

### Added

- Add helper functions for saving/loading fitted TLMCalibration objects
- Add better error message for non-try TLM methods

## [1.0.2] - 2025-02-14

### Added

- Add constraint outputs options to get_trustworthiness_score

## [1.0.1] - 2025-02-13

### Fixed

- Doc link in TLMOptions

### Added

- Release of the Cleanlab TLM Python client.

[Unreleased]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.5...HEAD
[1.1.5]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.23...v1.1.0
[1.0.23]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.22...v1.0.23
[1.0.22]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.21...v1.0.22
[1.0.21]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.20...v1.0.21
[1.0.20]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.19...v1.0.20
[1.0.19]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.18...v1.0.19
[1.0.18]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.17...v1.0.18
[1.0.17]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.16...v1.0.17
[1.0.16]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.15...v1.0.16
[1.0.15]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.14...v1.0.15
[1.0.14]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.13...v1.0.14
[1.0.13]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.12...v1.0.13
[1.0.12]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.11...v1.0.12
[1.0.11]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.10...v1.0.11
[1.0.10]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.9...v1.0.10
[1.0.9]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.7...v1.0.8
[1.0.6]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/cleanlab/cleanlab-tlm/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/cleanlab/cleanlab-tlm/releases/tag/v1.0.1
[1.0.0]: https://github.com/cleanlab/cleanlab-tlm/releases/tag/v1.0.0
