from llm_modelbench import doctor, hardware


def test_parse_nvidia_inventory_preserves_all_rows_and_order():
    output = "\n".join([
        "0, GPU-5060, 00000000:01:00.0, NVIDIA GeForce RTX 5060 Ti, 16311, 575.57, 12.0",
        "1, GPU-3060, 00000000:09:00.0, NVIDIA GeForce RTX 3060, 12288, 575.57, 8.6",
    ])

    devices = hardware._parse_nvidia_gpu_inventory(output)

    assert [device.physical_index for device in devices] == [0, 1]
    assert [device.name for device in devices] == [
        "NVIDIA GeForce RTX 5060 Ti", "NVIDIA GeForce RTX 3060",
    ]
    assert devices[0].uuid == "GPU-5060"
    assert devices[1].pci_bus_id == "00000000:09:00.0"
    assert devices[0].total_vram_mb == 16311.0
    assert devices[1].driver_version == "575.57"
    assert devices[1].compute_capability == "8.6"


def test_parse_nvidia_inventory_keeps_missing_optional_values_unavailable():
    output = "2, N/A, [N/A], NVIDIA Test GPU, N/A, , [Not Supported]\n"

    devices = hardware._parse_nvidia_gpu_inventory(output)

    assert devices == [hardware.GPUDevice(
        physical_index=2,
        uuid=None,
        pci_bus_id=None,
        name="NVIDIA Test GPU",
        total_vram_mb=None,
        driver_version=None,
        compute_capability=None,
    )]


def test_detect_gpus_retries_without_compute_capability(monkeypatch):
    calls = []
    monkeypatch.setattr(hardware.shutil, "which", lambda command: "/usr/bin/nvidia-smi" if command == "nvidia-smi" else None)

    def fake_run(command, timeout=5):
        calls.append(command)
        if "compute_cap" in command[1]:
            return None
        return "0, GPU-legacy, 00000000:01:00.0, NVIDIA Legacy, 8192, 535.54\n"

    monkeypatch.setattr(hardware, "_run", fake_run)

    assert hardware.detect_gpus() == [hardware.GPUDevice(
        physical_index=0,
        uuid="GPU-legacy",
        pci_bus_id="00000000:01:00.0",
        name="NVIDIA Legacy",
        total_vram_mb=8192.0,
        driver_version="535.54",
        compute_capability=None,
    )]
    assert len(calls) == 2


def test_detect_gpu_projects_first_inventory_device(monkeypatch):
    devices = [
        hardware.GPUDevice(4, "GPU-first", "00000000:01:00.0", "First", 16311.0, "575.57", "12.0"),
        hardware.GPUDevice(9, "GPU-second", "00000000:09:00.0", "Second", 12288.0, "575.57", "8.6"),
    ]
    monkeypatch.setattr(hardware, "detect_gpus", lambda: devices)

    assert hardware.detect_gpu() == hardware.GPUInfo(
        "nvidia", "First", 15.9, True, True, "575.57"
    )


def test_detect_gpus_and_compatibility_wrapper_report_no_gpu(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    monkeypatch.setattr(hardware, "_run", lambda command, timeout=5: None)

    assert hardware.detect_gpus() == []
    assert hardware.detect_gpu() == hardware.GPUInfo()


def test_doctor_render_exposes_inventory_without_changing_scalar_gpu_line():
    rendered = doctor.render({
        "llm_version": "test", "python": "3", "sys_executable": "python",
        "imported_from": "package", "entrypoint": None, "venv": None,
        "pythonpath": None, "ollama_url": "http://127.0.0.1:11434",
        "ollama_model_count": 0, "ollama_loaded_count": 0, "nvidia_smi": "nvidia-smi",
        "node": "node", "node_version": "v1", "gpu": {
            "vendor": "nvidia", "name": "First", "total_vram_gb": 15.9,
        },
        "gpus": [{
            "physical_index": 0, "name": "First", "uuid": "GPU-first",
            "pci_bus_id": "00000000:01:00.0", "total_vram_mb": 16311.0,
            "compute_capability": "12.0",
        }],
        "disk_free_gb": 1, "disk_total_gb": 2,
    })

    assert "GPU:               ok  nvidia  First  VRAM=15.9GB" in rendered
    assert "GPUs detected:     1 physical NVIDIA device(s)" in rendered
    assert "index=0 name=First uuid=GPU-first pci=00000000:01:00.0" in rendered
