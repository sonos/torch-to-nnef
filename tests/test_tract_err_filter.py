from torch_to_nnef.inference_target.tract import tract_err_filter


def test_filters_benign_mislabeled_symbol_warning():
    serr = (
        "[2026-09-01T09:10:48.591978000Z WARN  tract_nnef::deser] "
        "Assertion `TARGETS__TIME >= 1` constrains symbol(s) absent from "
        "every tensor shape (TARGETS__TIME); it has no effect. This "
        "usually means a mislabeled symbol name."
    )
    assert tract_err_filter(serr) == ""


def test_keeps_unrelated_warning():
    serr = "[... WARN tract_nnef::deser] some other unrelated warning"
    assert tract_err_filter(serr) == serr


def test_keeps_actual_error():
    serr = "[... ERROR tract_core] shape mismatch"
    assert tract_err_filter(serr) == serr
