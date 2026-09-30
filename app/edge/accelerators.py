"""ONNX Runtime sessions that target the Snapdragon Hexagon NPU first.

Execution-provider priority for device="auto":
    1. QNNExecutionProvider  — Hexagon NPU via the QNN HTP backend
                               (Snapdragon X PCs: `pip install onnxruntime-qnn`)
    2. DmlExecutionProvider  — GPU via DirectML (`onnxruntime-directml`)
    3. CPUExecutionProvider  — always available

The same code runs on an x86 dev laptop and a Snapdragon HP OmniBook; only
the installed onnxruntime package differs.
"""

import platform
from pathlib import Path

PROVIDER_LABELS = {
    "QNNExecutionProvider": "Snapdragon NPU (QNN HTP)",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CPUExecutionProvider": "CPU",
}

# Burst clocks for interactive latency; FP16 lets FP32 graphs run on the HTP.
QNN_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1",
}


def create_session(model_path: str | Path, device: str = "auto"):
    """Return (InferenceSession, active_provider). device: auto | npu | gpu | cpu."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list = []
    if device in ("auto", "npu") and "QNNExecutionProvider" in available:
        providers.append(("QNNExecutionProvider", QNN_OPTIONS))
    if device in ("auto", "gpu") and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    return session, session.get_providers()[0]


def system_info() -> dict:
    try:
        import onnxruntime as ort
        providers, version = ort.get_available_providers(), ort.__version__
    except ImportError:
        providers, version = [], None
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "onnxruntime": version,
        "available_providers": providers,
        "npu_available": "QNNExecutionProvider" in providers,
    }
