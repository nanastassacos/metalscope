import os
import tempfile

# Process-wide, set when anything imports the tests package and never undone: with no override, config.home()
# resolves to the real ~/.metalscope (which a running server may be using). Here, not only in conftest.py: a test
# module imported from outside tests/ (a scratch file's `from tests.test_x import *`) runs without conftest's
# fixtures. A per-test monkeypatch alone is not enough either: the trace writer thread reads the environment when
# it saves, which can be after a test's teardown restored it.
os.environ["METALSCOPE_HOME"] = tempfile.mkdtemp(prefix="metalscope-tests-")
