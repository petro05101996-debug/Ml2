import pandas as pd

import app


def test_build_target_selection_prefills_category_and_sku(monkeypatch):
    class _FakeSt:
        def selectbox(self, _label, options, **_kwargs):
            return options[0]

    monkeypatch.setattr(app, "st", _FakeSt())
    txn = pd.DataFrame(
        {
            "category": ["electronics", "electronics", "home"],
            "product_id": ["sku-2", "sku-1", "sku-3"],
        }
    )

    selection = app._build_target_selection(txn)

    assert selection["target_category"] == "electronics"
    assert selection["target_sku"] == "sku-1"
    assert selection["category_options"] == ["electronics", "home"]
    assert selection["sku_options"] == ["sku-1", "sku-2"]

