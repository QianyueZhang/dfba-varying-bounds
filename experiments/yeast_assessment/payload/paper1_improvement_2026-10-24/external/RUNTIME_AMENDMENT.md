# Runtime amendment: immutable result-array caching

The first 92-case execution was interrupted before any result file or scientific outcome was exposed. A process-status check had already recorded more than five minutes of CPU time, with the run still active. A one-second native stack sample showed repeated materialization of HiGHS basis-status vectors through pybind in SciPy 1.15.1 result postprocessing. The interrupted Python traceback was inside SciPy's `_highs_wrapper` after the ordinary `linprog` call.

Source inspection found that `basis.col_status[ii]` and `solution.col_dual[ii]` copy their complete bound vectors at each loop iteration. The new driver caches these two immutable vectors once after the optimizer has returned. It does not change a solver call, feasible set, objective, tolerance, parameter, case order or experiment selection. No installed dependency or frozen research file is edited. The protocol remains SHA-256 `3a57b04c332a36ce42c812b2266b23a8ccc41577d6f97b9106410d4dc16da445`.

Before rerunning the complete campaign, the original and cached wrappers solve the same first nominal LP. `runtime_equivalence.json` checks solver status, objective, full selected flux vector, equality/bound residuals and bound marginals for exact equality. Only after this check passes can `cached_scipy_driver.py --run` execute the unchanged numerical script. The driver records the original and modified wrapper-source hashes and its own hash.

The first interrupted run supplies no scientific result and is not pooled with the completed run. Runtime is not used as evidence of solver superiority. This amendment repairs interpreter/binding overhead only; mathematical claims remain those of the frozen protocol.
