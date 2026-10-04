import pytest

from pramaan.core.splits import SplitAccessError
from pramaan.improve.reranker_data import _assert_train_only


class _Store:
    def q(self, sql, params=None):
        return [{"filing_id": "dev-filing"}, {"filing_id": "test-filing"}]


ALL_PAGES = {
    "train-filing": ["train-filing#p1"],
    "dev-filing": ["dev-filing#p1"],
    "test-filing": ["test-filing#p1"],
}


def test_train_pages_pass():
    _assert_train_only(_Store(), ALL_PAGES, {"train-filing#p1"})


@pytest.mark.parametrize("page", ["dev-filing#p1", "test-filing#p1"])
def test_dev_or_test_pages_are_rejected(page):
    with pytest.raises(SplitAccessError):
        _assert_train_only(_Store(), ALL_PAGES, {"train-filing#p1", page})
