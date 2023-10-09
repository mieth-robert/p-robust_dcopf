# Data Valuation from Data-Driven Optimization

Code supplement to [Prescribed Robustness in Optimal Power Flow](https://arxiv.org/abs/2310.02957).

**Abstract:** 
For a timely decarbonization of our economy, power systems need to accommodate increasing numbers of clean but stochastic resources. This requires new operational methods that internalize this stochasticity to ensure safety and efficiency. This paper proposes a novel approach to compute adaptive safety intervals for each stochastic resource that internalize power flow physics and optimize the expected cost of system operations, making them ``prescriptive''. The resulting intervals are interpretable and can be used in a tractable robust optimal power flow problem as uncertainty sets. We use stochastic gradient descent with differentiable optimization layers to compute a mapping that obtains these intervals from a given vector of context parameters that captures the expected system state. We demonstrate and discuss the proposed approach on two case studies.

---

## Usage

Everything is implemented in Python and results reported in the paper can be reproduced by running the Jupyter notebooks.

---

## Citation
```
@article{mieth2023prescribed,
  title={Prescribed Robustness in Optimal Power Flow},
  author={Mieth, Robert and Poor, H Vincent},
  journal={arXiv preprint arXiv:2310.02957},
  year={2023}
}
```