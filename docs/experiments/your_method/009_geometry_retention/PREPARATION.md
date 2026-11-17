# Preparation record

No candidate training or experimental-seed rollout was performed during code
preparation. The initial six-test run had five passes and one failure: translating
the workspace changed a derived distance-progress feature by 1.49e-6 because
float32 target coordinates suffered cancellation. Relative geometry arithmetic
now uses float64 internally and exports float32 features. The tolerance was not
relaxed. The initial Ruff check passed.

Inspection also identified that functions imported through a dynamically named
helper cannot be submitted directly to Windows spawn workers. A top-level
importable wrapper is used; preflight exercises the actual evaluation wrapper.
Neither change is a model-performance tuning trial.

Final verification: 12 tests passed in 9.05 seconds (new recovery-policy tests
plus the existing training tests). Preflight preserved all 454 actions across
the ten historical successful training scenes, checked all 1440 original rows
produce finite 381-dimensional features, and completed a frozen two-step
synthetic evaluation through a Windows process worker. Its trace and checkpoint
are under `preparation/`; this is not a performance experiment. Final Ruff check
passed after arranging the preflight imports. No experimental training had run
when this protocol was committed.
