import json
import pytest

from llm_modelbench import runner
from llm_modelbench.config import Config
from llm_modelbench.ollama import MockClient
from llm_modelbench.runtime_identity import RuntimeIdentity, RuntimeModelIdentity, RuntimeExecutionSettings

MODEL="qwen2.5-coder:14b"; U="GPU-00000000-0000-0000-0000-000000000001"
def identity(**change):
    value=dict(backend="ollama",adapter_identity="MockClient",endpoint="http://127.0.0.1:11434",profile_name="fixture",profile_provenance="test",profile_schema_version=1,server_version="fixture",model=RuntimeModelIdentity(MODEL,MODEL,"sha256:one",provenance="test"),physical_gpu_uuids=(U,),declared_device_order=(U,),execution=RuntimeExecutionSettings("single_device",context_size=1024),evidence_provenance="test")
    value.update(change); return RuntimeIdentity(**value)
class Client(MockClient):
    def __init__(self): super().__init__(); self.calls=0
    def chat(self,*a,**k): self.calls+=1; return super().chat(*a,**k)
def run(c,path,identity_value,resume):
    return runner.run(c,Config(fingerprint=False),level="smoke",out_dir=path,include=None,exclude=None,skip_offload=False,categories=None,task_ids=["py_anagram"],resume=resume,live_ui="off",fingerprint_enabled=False,selected_models=[MODEL],capability_profiles={MODEL:{"declared_capabilities":["completion"],"supported_families":["text"]}},auto_probe=False,runtime_identity={MODEL:identity_value})
def test_new_run_writes_model_keyed_identity_and_row_reference(tmp_path):
    run(Client(),tmp_path,identity(),False); artifact=json.loads((tmp_path/"runtime_identity.json").read_text()); row=json.loads((tmp_path/"raw_results.jsonl").read_text().splitlines()[0])
    assert artifact["identities"][MODEL]["model"]["artifact_digest"]=="sha256:one" and row["runtime_identity_hash"]==artifact["identities"][MODEL]["identity_hash"]
def test_compatible_resume_reuses_without_rewriting_identity(tmp_path):
    run(Client(),tmp_path,identity(),False); before=(tmp_path/"runtime_identity.json").read_bytes(); c=Client();run(c,tmp_path,identity(),True)
    assert c.calls==0 and (tmp_path/"runtime_identity.json").read_bytes()==before
@pytest.mark.parametrize("changed,code",[(dict(endpoint="http://127.0.0.1:11435"),"endpoint_changed"),(dict(model=RuntimeModelIdentity(MODEL,MODEL,"sha256:two",provenance="test")),"model_artifact_changed"),(dict(execution=RuntimeExecutionSettings("single_device",context_size=2048)),"context_changed")])
def test_mismatch_refuses_before_rows_or_tasks(tmp_path,changed,code):
    run(Client(),tmp_path,identity(),False); before={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()};c=Client()
    with pytest.raises(ValueError,match=code): run(c,tmp_path,identity(**changed),True)
    assert c.calls==0 and before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
def test_legacy_identity_refuses(tmp_path):
    (tmp_path/"raw_results.jsonl").write_text('{"model":"qwen2.5-coder:14b","task":"py_anagram","task_hash":"x"}\n'); before=(tmp_path/"raw_results.jsonl").read_bytes()
    with pytest.raises(ValueError,match="legacy_runtime_identity_missing"):run(Client(),tmp_path,identity(),True)
    assert (tmp_path/"raw_results.jsonl").read_bytes()==before
def test_malformed_identity_refuses(tmp_path):
    (tmp_path/"raw_results.jsonl").write_text('{"model":"qwen2.5-coder:14b","task":"py_anagram","task_hash":"x"}\n');(tmp_path/"runtime_identity.json").write_text('{bad'); before=(tmp_path/"runtime_identity.json").read_bytes()
    with pytest.raises(ValueError,match="runtime_identity_artifact_unavailable"):run(Client(),tmp_path,identity(),True)
    assert (tmp_path/"runtime_identity.json").read_bytes()==before
