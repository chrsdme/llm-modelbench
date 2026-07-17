import pytest
from llm_modelbench.dossier import composite_score, validate_weights

class T:
    def __init__(self,id,category): self.id,self.category=id,category

def test_dossier_renormalizes_and_rejects_bad_weights():
    tasks=[T("a","ocr"),T("b","coding_python")]
    ledger={"d":{"categories":{"ocr":{"task_ids_covered":["a"]}}}}
    result=composite_score("d",ledger,{"ocr":80},{"ocr":.5,"coding_python":.5},tasks)
    assert result["composite"] == 80 and result["pending_categories"] == ["coding_python"]
    with pytest.raises(ValueError): validate_weights({"ocr":.9})
