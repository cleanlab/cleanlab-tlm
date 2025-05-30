# Cleanlab Trustworthy Language Model (TLM) - Reliability and explainability added to every LLM output

[![Build Status](https://github.com/cleanlab/cleanlab-tlm/actions/workflows/ci.yml/badge.svg)](https://github.com/cleanlab/cleanlab-tlm/actions/workflows/ci.yml) [![PyPI - Version](https://img.shields.io/pypi/v/cleanlab-tlm.svg)](https://pypi.org/project/cleanlab-tlm) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cleanlab-tlm.svg)](https://pypi.org/project/cleanlab-tlm)

In one line of code, Cleanlab TLM adds real-time evaluation of every response in GenAI, RAG, LLM, and Agent systems.

## Setup

This tutorial requires a TLM API key. Get one [here](https://tlm.cleanlab.ai/).

```console
export CLEANLAB_TLM_API_KEY=<YOUR_API_KEY_HERE>
```

Install the package:

```console
pip install cleanlab-tlm
```

## Usage

To get started, copy the code below to try your own prompt or score existing prompt/response pairs with ease.

```python
from cleanlab_tlm import TLM
tlm = TLM(options={"log": ["explanation"], "model": "gpt-4.1-mini"}) # GPT, Claude, etc.
out = tlm.prompt("What's the third month of the year alphabetically?")
print(out)
```

TLM returns a dictionary containing `response`, `trustworthiness_score`, and any requested optional fields like `explanation`.

```json
{
  "response": "March.",
  "trustworthiness_score": 0.4590804375945598,
  "explanation": "Found an alternate response: December"
}
```

## Why TLM?

- **Trustworthiness Scores**: Each response comes with a trustworthiness score, helping you [reliably](https://cleanlab.ai/blog/trustworthy-language-model/) gauge the likelihood of hallucinations.
- **Higher accuracy**: Rigorous [benchmarks](https://cleanlab.ai/blog/trustworthy-language-model/) show TLM consistently produces more accurate results than other LLMs like o3/o1, GPT 4o, and Claude.
- **Scalable API**: Designed to handle large datasets, TLM is suitable for most enterprise applications, including data extraction, tagging/labeling, Q&A (RAG), and more.

## Documentation

Comprehensive documentation along with tutorials and examples can be found [here](https://help.cleanlab.ai/tlm).

## License

`cleanlab-tlm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
