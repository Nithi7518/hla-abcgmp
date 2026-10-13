import os

# Tests must be deterministic and offline: never start a local or cloud LLM from the test suite.
os.environ["LLM_DISABLE"] = "1"
