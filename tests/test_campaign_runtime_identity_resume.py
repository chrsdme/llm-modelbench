"""Isolated campaign identity-map gates; no client calls or campaign mutation."""
import copy
import pytest
from llm_modelbench.runtime_identity import RuntimeIdentity, RuntimeModelIdentity, RuntimeExecutionSettings, validate_frozen_runtime_identity_map

U1="GPU-00000000-0000-0000-0000-000000000001"; U2="GPU-00000000-0000-0000-0000-000000000002"
def identity(display,digest,**changes):
    v=dict(backend="ollama",adapter_identity="Fake",endpoint="http://127.0.0.1:11434",profile_name="p",profile_provenance="fixture",profile_schema_version=1,server_version="v",model=RuntimeModelIdentity(display,display,digest,provenance="fixture"),physical_gpu_uuids=(U1,U2),declared_device_order=(U1,U2),execution=RuntimeExecutionSettings("layer_split",{U1:1,U2:1},context_size=1024),evidence_provenance="fixture");v.update(changes);return RuntimeIdentity(**v)
def maps():
    current={"a":identity("a","sha256:a"),"b":identity("b","sha256:b")};return {k:v.to_dict() for k,v in current.items()},current
def test_campaign_plan_identity_map_preserves_per_model_digests():
    frozen,_=maps();assert frozen["a"]["model"]["artifact_digest"] != frozen["b"]["model"]["artifact_digest"]
def test_compatible_two_model_campaign_map():
    frozen,current=maps();assert validate_frozen_runtime_identity_map(frozen,current,["a","b"]).compatible
@pytest.mark.parametrize("change,code",[
    (dict(endpoint="http://127.0.0.1:11435"),"endpoint_changed"),
    (dict(physical_gpu_uuids=(U1,),declared_device_order=(U1,),execution=RuntimeExecutionSettings("single_device",context_size=1024)),"physical_gpu_uuids_changed"),
    (dict(declared_device_order=(U2,U1)),"device_order_changed"),
    (dict(model=RuntimeModelIdentity("a","a","sha256:x",provenance="fixture")),"model_artifact_changed"),
    (dict(execution=RuntimeExecutionSettings("layer_split",{U1:1,U2:1},context_size=2048)),"context_changed"),
])
def test_campaign_mismatch_codes(change,code):
    frozen,current=maps();current["a"]=identity("a","sha256:a",**change)
    with pytest.raises(ValueError,match=code):validate_frozen_runtime_identity_map(frozen,current,["a","b"])
def test_missing_swapped_legacy_malformed_schema_and_hash_refuse():
    frozen,current=maps()
    for value,code in [({"a":frozen["a"]},"runtime_identity_model_missing"),({"a":frozen["b"],"b":frozen["a"]},"model_artifact_changed"),(None,"legacy_runtime_identity_missing"),({"a":{}},"runtime_identity_artifact_unavailable")]:
        with pytest.raises(ValueError,match=code):validate_frozen_runtime_identity_map(value,current,["a","b"])
    bad=copy.deepcopy(frozen);bad["a"]["schema_version"]=999
    with pytest.raises(ValueError,match="runtime_identity_artifact_unavailable"):validate_frozen_runtime_identity_map(bad,current,["a","b"])
    bad=copy.deepcopy(frozen);bad["a"]["identity_hash"]="0"*64
    with pytest.raises(ValueError,match="runtime_identity_artifact_unavailable"):validate_frozen_runtime_identity_map(bad,current,["a","b"])
