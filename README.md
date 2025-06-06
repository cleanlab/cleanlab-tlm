# Cleanlab Trustworthy Language Model (TLM) - Trust Scores for every LLM output

[![Build Status](https://github.com/cleanlab/cleanlab-tlm/actions/workflows/ci.yml/badge.svg)](https://github.com/cleanlab/cleanlab-tlm/actions/workflows/ci.yml) [![PyPI - Version](https://img.shields.io/pypi/v/cleanlab-tlm.svg)](https://pypi.org/project/cleanlab-tlm) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cleanlab-tlm.svg)](https://pypi.org/project/cleanlab-tlm)

In one line of code, Cleanlab TLM adds real-time evaluation of every response in LLM, RAG, and Agent systems.

## Setup

TLM requires an API key. Get one [here](https://tlm.cleanlab.ai/) for free.

```console
export CLEANLAB_TLM_API_KEY=<YOUR_API_KEY_HERE>
```

Install the package:

```console
pip install cleanlab-tlm
```

## Usage

TLM automatically scores the trustworthiness of responses generated from your own LLM in real-time:

```python
from cleanlab_tlm import TLM

tlm = TLM(options={"log": ["explanation"]})
tlm.get_trustworthiness_score(
    prompt="What's the third month of the year alphabetically?",
    response="August"  # generated from any LLM model using the same prompt
)
```

This returns a dictionary with `trustworthiness_score` and optionally requested fields like `explanation`.

```json
{
  "trustworthiness_score": 0.02993446111679077,
  "explanation": "Found alternate plausible response: December"
}
```


Alternatively, you generate responses and simultaneously score them with TLM:

```python
tlm = TLM(options={"log": ["explanation"], "model": "gpt-4.1-mini"})  # GPT, Claude, etc.
tlm.prompt("What's the third month of the year alphabetically?")
```

This additionally returns a `response`.

```json
{
  "response": "March.",
  "trustworthiness_score": 0.4590804375945598,
  "explanation": "Found alternate plausible response: December"
}
```

## Why TLM?

- **Trustworthiness Scores**: Every LLM response is scored via [state-of-the-art](https://cleanlab.ai/blog/trustworthy-language-model/) uncertainty estimation, helping you reliably gauge the likelihood of hallucinated/incorrect responses.
- **Higher accuracy**: Rigorous [benchmarks](https://cleanlab.ai/blog/trustworthy-language-model/) show TLM consistently produces more accurate scores than other hallucination detectors and responses than other LLMs.
- **Scalable API**: TLM is suitable for all enterprise applications where correct LLM responses are vital, including data extraction, tagging/labeling, Q&A (RAG), Agents, and more.

## Documentation

Comprehensive documentation and tutorials can be found [here](https://help.cleanlab.ai/tlm).

## License

`cleanlab-tlm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
