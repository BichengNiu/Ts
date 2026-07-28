"""Public API tests for structural-break and stability methods."""

import Ts
import Ts.TsTests as tests_api


STRUCTURAL_BREAK_EXPORTS = (
    "BaiPerronTest",
    "BaiPerronTestResult",
    "ChowTest",
    "ChowTestResult",
    "CUSUMTest",
    "CUSUMTestResult",
    "LeeStrazicichTwoBreakTest",
    "LeeStrazicichTwoBreakTestResult",
)


def test_structural_break_names_are_exported_from_both_public_namespaces():
    for name in STRUCTURAL_BREAK_EXPORTS:
        assert name in tests_api.__all__
        assert name in Ts.__all__
        assert getattr(tests_api, name) is getattr(Ts, name)
