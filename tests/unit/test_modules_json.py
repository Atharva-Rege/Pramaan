import pytest

from pramaan.agent.modules import ModuleParseError, extract_json


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2,]}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here it is: {"steps": []} hope it helps') == {"steps": []}


def test_extract_json_failure():
    with pytest.raises(ModuleParseError):
        extract_json("no json here")
