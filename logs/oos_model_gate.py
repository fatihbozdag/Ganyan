"""Compatibility entry point for the strict packaged accuracy gate.

Requires explicit --from, --to and --output. Retrospective runs cannot promote.
"""
from ganyan.predictor.ml.gate import assert_min_window, load_named, mcnemar_exact, main

if __name__ == "__main__":
    main()
