"""TaskBench SFT target-format comparison.

A reproducible experiment harness for comparing two supervised fine-tuning
target formats on Microsoft JARVIS / TaskBench data:

* **Mode A (full_json)** – the model emits the complete TaskBench plan object
  (``task_steps`` / ``task_nodes`` / ``task_links``).
* **Mode B (trajectory)** – the model emits only the ordered list of tool IDs.

The official TaskBench gold schema and evaluation logic are *reused* (see
:mod:`taskbench_sft.official`); we never re-implement or mutate the gold labels.
"""

__version__ = "0.1.0"
