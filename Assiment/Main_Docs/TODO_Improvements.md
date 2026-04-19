# TODO Improvements

## Engineering problem framing

- Add a stronger real-world deployment description for the inspection station: camera position, lighting assumptions, cycle time, and operator workflow.
- Clarify whether the inspected object is always a cup surface and whether the target defect types are limited to visible deformation and spotting, or include a broader defect family.
- Tighten the engineering impact statement with a more concrete argument about why false PASS decisions matter more than false FAIL decisions in this setting.
- Add a short paragraph explaining the decision threshold and how PASS/FAIL would be integrated into a downstream reject mechanism.

## Dataset provenance and consistency

- Document exactly how `runs/data_balanced` was produced and which raw images were used to create that balanced experiment snapshot.
- Explain why the repository now contains two dataset states with different class names: `defect/non_defect` and `defective/non_defective`.
- Confirm whether `zeroq_cup_classification_scaffold/data/processed` is intended to replace the balanced experiment snapshot or whether it is only a later rerun dataset.
- Add a short provenance note for augmentation and balancing so the reader can tell which results came from raw captures, offline augmentation, or later processed splits.

## Result reporting quality

- Persist a full YOLO test classification report with precision, recall, F1, and confusion matrix instead of relying only on stored top-1 and top-5 summary metrics.
- Decide whether YOLO should remain in the same aggregate ranking chart when its `f1` entry is only a mirrored top-1 value in `all_results.csv`.
- Add one cleaner final summary table with consistent metric semantics across all model families.
- Review whether the scratch CNN section should explicitly state that it satisfies the assignment requirement even though it is not the strongest model.

## Demonstration evidence

- Insert a real inspection-setup image or screenshot instead of relying on inherited scaffold references.
- Prepare at least five genuinely unseen demonstration samples and document their predicted PASS/FAIL outputs.
- Save one or two live inference screenshots from the current preferred model so the report and demo tie together more clearly.
- Confirm which script will be used in the live demonstration path and document the exact command.

## Document polish

- Add figure numbering/captions in a more presentation-ready form if the final submission needs tighter formatting than Markdown-to-docx provides by default.
- Review paragraph length and compress some narrative if the final report must fit a strict page target.
- Recheck wording around "best overall" so the report stays precise about YOLO top-1 versus full binary F1.
- If time allows, add a short appendix or note listing the exact run folders used as the source of truth.
