"""Explicit approved heads and content-addressed, gate-checked promotion."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile


def model_root():
    return Path(os.environ.get("GANYAN_MODEL_DIR", Path.cwd() / "models")).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def candidate_directory(directory=None):
    directory = Path(directory).resolve() if directory else model_root() / "candidates"
    if directory == model_root() or (model_root() / "releases") in directory.parents:
        raise ValueError("Training must use a candidate directory, not production artifacts")
    return directory


def approved_manifest(root=None):
    root = Path(root or model_root())
    path = root / "active.json"
    if not path.exists():
        path = Path(__file__).with_name("approved_models.json")
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or not manifest.get("heads"):
        raise ValueError("Invalid approved model manifest")
    return manifest


def approved_paths(name, root=None):
    root = Path(root or model_root())
    record = approved_manifest(root)["heads"][name]
    paths = {}
    for kind in ("model", "metadata"):
        path = (root / record[kind]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Model manifest path escapes model directory")
        if digest(path) != record[kind + "_sha256"]:
            raise ValueError(f"Approved artifact changed: {name} {kind}")
        paths[kind] = path
    return paths


def artifact_identity(model_path, meta_path):
    return {"model_sha256": digest(model_path), "metadata_sha256": digest(meta_path)}


def promote(candidate, gate_path, *, head="lightgbm_ranker", root=None):
    """Serialize promotions so concurrent gates cannot overwrite one another."""
    root = Path(root or model_root())
    lock = root / ".promotion.lock"
    try:
        stream = lock.open("x")
    except FileExistsError as exc:
        raise ValueError("Another promotion is in progress; inspect .promotion.lock") from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
        return _promote_locked(candidate, gate_path, head=head, root=root)
    finally:
        lock.unlink(missing_ok=True)


def _promote_locked(candidate, gate_path, *, head="lightgbm_ranker", root=None):
    """Install immutable files, then switch one manifest atomically. Never rewrite weights."""
    root = Path(root or model_root())
    candidate = Path(candidate)
    model, meta = candidate.with_suffix(".txt"), candidate.with_suffix(".meta.json")
    gate = json.loads(Path(gate_path).read_text())
    identity = artifact_identity(model, meta)
    current = approved_manifest(root)
    live = approved_paths(head, root)
    live_id = artifact_identity(live["model"], live["metadata"])
    if (gate.get("version") != 2 or gate.get("swap") is not True
            or gate.get("mode") != "forward"
            or gate.get("candidate_artifact") != identity
            or gate.get("live_artifact") != live_id
            or gate.get("pipeline_sha256") != pipeline_digest()
            or gate.get("n", 0) < 1500 or gate.get("coverage_days", 0) < 365
            or gate.get("delta_pp", 0) < 1.0
            or not 0 <= gate.get("mcnemar_p", 1.0) < 0.05):
        raise ValueError("Promotion requires a current, passing forward gate for these exact artifacts")
    release = root / "releases" / (identity["model_sha256"] + "-" + identity["metadata_sha256"])
    release.mkdir(parents=True, exist_ok=True)
    for source, filename in ((model, "model.txt"), (meta, "model.meta.json")):
        target = release / filename
        if target.exists() and digest(target) != digest(source):
            raise ValueError("Immutable release conflict")
        if not target.exists():
            shutil.copyfile(source, target)
    if artifact_identity(release / "model.txt", release / "model.meta.json") != identity:
        raise ValueError("Candidate changed while promotion was copying it")
    current["heads"][head] = {
        "model": str((release / "model.txt").relative_to(root)),
        "metadata": str((release / "model.meta.json").relative_to(root)), **identity,
    }
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".active-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(current, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "active.json")
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return release
