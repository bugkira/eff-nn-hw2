# pararnn-torch (Efficient NN HW-2 subset)

Homework subset of [`bugkira/pararnn-torch`](https://github.com/bugkira/pararnn-torch) for **Efficient Neural Networks · ДЗ-2**.

Public surface kept for the talk / reproduction:

- `ParaGRU`, `ParaLSTM`
- `ParaRNN`, `newton_apply`, `sequential_apply`, `verify_agreement`
- decode helpers (`decode_step`, …)

Other cells (sLSTM, CfC, NLRU, …) are stubbed out in this fork and raise `NotImplementedError` if constructed.

```bash
pip install -e .
python examples/parity.py
```

Paper: Danieli et al., *ParaRNN*, ICLR 2026 · arXiv:2510.21450  
Upstream library: https://github.com/bugkira/pararnn-torch
