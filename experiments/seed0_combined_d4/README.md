# Combined D4-conv + D4 direction sharing, seed0

ONE model uses shift_implementation=d4_tied_conv AND
 direction_share_implementation=d4. This is a new combined ablation, not an
ensemble, not two independent variants, and not fused up/down communication.
Both operators are learned jointly at all four depths. Hard cummax, fused
softmax, reused/fused-normalisation sharing, fused Adam, direct checks and explicit
runtime dispatch retain the established optimised settings.

40 fixed training tasks, pilot20 + additional20, 1500 steps, seed0. One array
0-39%2, one GPU per element, at most two concurrent. No model weights/checkpoints
are saved. Normal W&B metrics, candidate tables, effective config and code commit
are retained under variant combined-d4-conv-sharing and tag seed0-combined-d4.
This is separate from the single-operator seed0 results.

From repository root:

    sbatch experiments/seed0_combined_d4/submit.slurm

Dry-run one task:

    python experiments/seed0_combined_d4/run_job.py --index 0 --dry-run

Inside an allocated GPU, run the combined architecture correctness test first:

    python -m pytest src/tests/correctness/training/test_combined_d4_model.py -q -x

Then omit --dry-run to execute just index0 if a one-task check is desired.
Do not also submit the two independent retest arrays unless those additional
experiments are intended. No full-model D4-equivariance claim is established by
selecting these two operators. Accuracy and runtime of the combination must be
measured; they cannot be inferred by adding the separate variants' results.
