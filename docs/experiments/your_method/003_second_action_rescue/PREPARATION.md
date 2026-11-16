# Preparation checks

All eight audit tests passed on the first invocation. The tests verify one shared
correction plus one treatment-only correction, and reject missing shared actions,
additional later overrides and changed fork inputs. Ruff passed.

Both historical second disagreements were verified against the original
demonstrations and attempt-002 corrected traces. The full targeted snapshot
restored exactly and remained unchanged through the freeze check. These checks
used zero simulator actions and zero optimizer updates. The evidence output
directory did not exist before the planned single execution.
