"""What provisioning is allowed to record: the kind of a file, what the source says about
its bytes, and the path an asset may serve.

No fetch, no extraction, no manifest in a real root, no model: the unit under test is
the decisions the tool encodes - `classify_asset`, the published-digest comparison, the
scope contract and the review state - and all are exercised against synthetic bytes in a
temporary directory.

The scope flag exists because an asset downloaded to measure something is not thereby
licensed, reviewed or fit to be served by the shipped voice path, and the old tool
recorded none of that distinction. The kind table exists because the old map labelled
everything that was not a weight a tokenizer asset, which put a model's own settings
and a licence file under a vocabulary's name.

The digest comparison exists because the tool used to write "upstream publishes no
digest for this artefact" about a Hugging Face snapshot whose published digests were
sitting in the local cache the whole time - an unconditional sentence that let a
mismatch pass as a note. Approval moved the other way: supplying ``--approved-by`` used
to be enough to write ``approved`` into every manifest, so the licence gate's central
fact was produced by a tool instead of granted by a person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nativevoice.config.paths import hf_hub_cache_root, measurement_models_dir, models_dir  # noqa: E402
from nativevoice.core.errors import ConfigurationError  # noqa: E402
from nativevoice.models.api import AssetKind, AssetScope, ReviewStatus  # noqa: E402
from nativevoice.models.manifest_io import from_dict, to_dict  # noqa: E402
from nativevoice.stt.adapters.faster_whisper_engine import default_cache_root  # noqa: E402
from tools.provision import (  # noqa: E402
    CATALOGUE,
    build_manifest,
    build_parser,
    check_published_digest,
    check_scope_contract,
    classify_asset,
    default_registry_root,
    digest_notice,
    git_blob_sha1,
    hf_snapshot_dir,
    published_digests,
    resolve_registry_root,
    review_state,
    sha256,
    snapshot_files,
    tree_key,
)

BLOB = b"synthetic-asset-bytes-not-a-real-model"
TEACHER = "faster-whisper-large-v3"
TEACHER_REPO = "Systran/faster-whisper-large-v3"
PIN = "edaa852ec7e145841d8ffdb056a99866b5f0a478"


def make_asset_dir(tmp_path: Path, names: list[str]) -> tuple[Path, list[Path]]:
    """A package-shaped directory of placeholder files, nothing downloaded."""
    directory = tmp_path / "snapshots" / "deadbeef"
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BLOB)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    return directory, files


def provision_args(scope: str, **overrides) -> argparse.Namespace:
    base = dict(
        scope=scope,
        language=None,
        approved_by="tester@example.com",
        authority="quoted instruction",
        grant_product_approval=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)



def run(argv: list[str]):
    return build_parser().parse_args(argv)


BASE_ARGS = ["--asset", "faster-whisper-base", "--approved-by", "tester", "--authority", "quoted instruction"]


# --- asset kinds ---------------------------------------------------------------

def test_weights_model_configuration_and_vocabulary_are_three_different_kinds(tmp_path):
    directory, files = make_asset_dir(
        tmp_path, ["config.json", "preprocessor_config.json", "model.bin", "model.int8.onnx", "tokenizer.json", "vocabulary.txt"]
    )
    kinds = {path.name: classify_asset(path, directory) for path in files}
    assert kinds["model.bin"] is AssetKind.MODEL
    assert kinds["model.int8.onnx"] is AssetKind.MODEL
    assert kinds["config.json"] is AssetKind.MODEL_CONFIG
    assert kinds["preprocessor_config.json"] is AssetKind.MODEL_CONFIG
    assert kinds["tokenizer.json"] is AssetKind.TOKENIZER
    assert kinds["vocabulary.txt"] is AssetKind.TOKENIZER


def test_model_configuration_is_recorded_by_its_own_name(tmp_path):
    """The serialized value is a contract a human edits, so it is pinned as a string."""
    directory, files = make_asset_dir(tmp_path, ["config.json"])
    assert classify_asset(files[0], directory).value == "model_config"
    # A file that merely ends in `.json` is not model configuration.
    other, other_files = make_asset_dir(tmp_path / "b", ["tokenizer.json"])
    assert classify_asset(other_files[0], other) is AssetKind.TOKENIZER


def test_sherpa_text_front_end_files_keep_their_kind(tmp_path):
    """Every file the installed TTS package contributes, classified by what it is."""
    directory, files = make_asset_dir(
        tmp_path,
        [
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "lexicon-zh.txt",
            "date-zh.fst",
            "dict/jieba.dict.utf8",
            "dict/pos_dict/char_state_tab.utf8",
            "espeak-ng-data/af_dict",
            "espeak-ng-data/lang/an/afrikaans",
        ],
    )
    kinds = {path.relative_to(directory).as_posix(): classify_asset(path, directory) for path in files}
    assert kinds["model.int8.onnx"] is AssetKind.MODEL
    assert kinds["voices.bin"] is AssetKind.MODEL
    assert all(kind is AssetKind.TOKENIZER for name, kind in kinds.items() if name not in {"model.int8.onnx", "voices.bin"})


def test_the_package_files_that_used_to_be_refused_each_get_one_exact_kind(tmp_path):
    """README, LICENSE, `.gitattributes` and a bundled script, named one by one.

    These are the four that stopped the installed TTS package from being recorded. Each
    is now decided by its own name, not by a suffix skip-list that would silently drop
    anything else shaped like it: a licence is not documentation, documentation is not a
    tokenizer asset, and a Python script that ships inside a data package is code whose
    licence is a question of its own.
    """
    directory, files = make_asset_dir(
        tmp_path, ["LICENSE", "README.md", ".gitattributes", "dict/generate_user_dict.py"]
    )
    kinds = {path.name: classify_asset(path, directory) for path in files}
    assert kinds["LICENSE"] is AssetKind.LICENCE
    assert kinds["README.md"] is AssetKind.DOCUMENTATION
    assert kinds[".gitattributes"] is AssetKind.PACKAGE_METADATA
    assert kinds["generate_user_dict.py"] is AssetKind.SOURCE
    # Recorded by name, so the serialised value is stable evidence rather than an enum's position.
    assert {name: kind.value for name, kind in kinds.items()} == {
        "LICENSE": "licence",
        "README.md": "documentation",
        ".gitattributes": "package_metadata",
        "generate_user_dict.py": "source",
    }


def test_a_licence_and_a_model_card_are_not_the_same_kind(tmp_path):
    """Prose that ships *as terms* and prose that ships *about* the model are different records."""
    directory, files = make_asset_dir(tmp_path, ["LICENSE", "LICENSE.txt", "README.md", "COPYING"])
    kinds = {path.name: classify_asset(path, directory) for path in files}
    assert kinds["LICENSE"] is kinds["COPYING"] is kinds["LICENSE.txt"] is AssetKind.LICENCE
    assert kinds["README.md"] is AssetKind.DOCUMENTATION
    # A file merely *named* like prose in another language stays a question, not a guess.
    other, other_files = make_asset_dir(tmp_path / "b", ["license_zh.txt"])
    with pytest.raises(ConfigurationError):
        classify_asset(other_files[0], other)


def test_an_unrecognised_file_is_refused_rather_than_labelled(tmp_path):
    """Nothing is skipped and nothing is defaulted: an unknown file is a decision to make.

    A broad suffix skip-list would have recorded a manifest with these files missing from
    it, which excludes them from every later integrity check - the same defect the
    refusal exists to prevent.
    """
    directory, files = make_asset_dir(tmp_path, ["notes.txt", "summary.csv", "metadata.yaml"])
    for path in files:
        with pytest.raises(ConfigurationError) as exc:
            classify_asset(path, directory)
        # The message names the file, so the operator knows what to decide about.
        assert path.name in str(exc.value)


def test_one_unrecognised_file_stops_the_whole_asset_being_written(tmp_path):
    """Partial provenance is worse than none: a manifest missing a digest silently
    excludes that file from every later integrity check."""
    directory, files = make_asset_dir(tmp_path, ["model.bin", "config.json", "tokenizer.json", "notes.txt"])
    with pytest.raises(ConfigurationError) as exc:
        build_manifest("faster-whisper-base", directory, files, provision_args("product"))
    assert "notes.txt" in str(exc.value)


# --- scope contract ------------------------------------------------------------

def test_scope_choices_are_exactly_the_documented_asset_scopes():
    parser = build_parser()
    scope = next(action for action in parser._actions if action.dest == "scope")
    assert scope.required is True
    assert set(scope.choices) == {value.value for value in AssetScope}


def test_omitting_the_scope_is_an_error_and_not_a_quiet_product_default():
    with pytest.raises(SystemExit):
        run(BASE_ARGS)


@pytest.mark.parametrize("bad", ["prod", "PRODUCT", "measurement ", "benchmarking", ""])
def test_an_invalid_scope_is_rejected_not_fallen_back_from(bad):
    with pytest.raises(SystemExit):
        run(BASE_ARGS + ["--scope", bad])


def test_a_measurement_asset_serialises_measurement_and_survives_the_round_trip(tmp_path):
    directory, files = make_asset_dir(tmp_path, ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"])
    manifest = build_manifest("faster-whisper-base", directory, files, provision_args("measurement"))
    assert manifest.use_scope is AssetScope.MEASUREMENT
    data = to_dict(manifest)
    assert data["use_scope"] == "measurement"
    assert from_dict(data).use_scope is AssetScope.MEASUREMENT
    assert from_dict(data) == manifest
    assert "measurement path" in manifest.notes


def test_product_scope_is_recorded_only_because_it_was_asked_for(tmp_path):
    directory, files = make_asset_dir(tmp_path, ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"])
    manifest = build_manifest("faster-whisper-base", directory, files, provision_args("product"))
    assert manifest.use_scope is AssetScope.PRODUCT
    assert to_dict(manifest)["use_scope"] == "product"
    # Recording the product path is not a rights claim: nothing here is redistributable.
    assert manifest.provenance.redistributable is False
    assert "routing permission, not a rights clearance" in manifest.notes


def test_a_scope_string_the_schema_cannot_hold_raises_instead_of_becoming_product(tmp_path):
    directory, files = make_asset_dir(tmp_path, ["model.bin"])
    with pytest.raises(ValueError):
        build_manifest("faster-whisper-base", directory, files, provision_args("benchmarking"))


# --- registry roots ------------------------------------------------------------

def test_each_scope_gets_its_own_default_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    assert default_registry_root(AssetScope.PRODUCT) == models_dir() == tmp_path / "models"
    assert default_registry_root(AssetScope.MEASUREMENT) == measurement_models_dir()
    assert resolve_registry_root(AssetScope.PRODUCT) != resolve_registry_root(AssetScope.MEASUREMENT)
    assert not measurement_models_dir().exists()  # choosing a root does not create it


def test_an_explicit_root_inside_the_other_scopes_tree_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path))
    shipped = default_registry_root(AssetScope.PRODUCT)
    with pytest.raises(ConfigurationError):
        resolve_registry_root(AssetScope.MEASUREMENT, str(shipped))
    with pytest.raises(ConfigurationError):
        resolve_registry_root(AssetScope.PRODUCT, str(default_registry_root(AssetScope.MEASUREMENT)))
    # The whole product data tree counts as overlap, not only its models directory:
    # a registry also discovers `<root>/*.json`, so a parent would swallow both.
    with pytest.raises(ConfigurationError):
        resolve_registry_root(AssetScope.MEASUREMENT, str(tmp_path))


def test_an_unrelated_explicit_root_is_accepted_for_either_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "configured"))
    trial = tmp_path / "trial-root"
    assert resolve_registry_root(AssetScope.MEASUREMENT, str(trial)).name == "trial-root"
    assert resolve_registry_root(AssetScope.PRODUCT, str(trial)).name == "trial-root"
    assert not trial.exists()


def test_the_root_it_returns_is_the_resolved_one(tmp_path, monkeypatch):
    """An unresolved root is two answers to "where did this asset go", and the audit
    then checks a path that is not the one provisioning wrote to."""
    monkeypatch.setenv("NATIVEVOICE_HOME", str(tmp_path / "configured"))
    trial = tmp_path / "trial-root"
    given = trial / ".." / "trial-root" / "manifests" / ".."
    resolved = resolve_registry_root(AssetScope.MEASUREMENT, str(given))
    assert resolved == trial.resolve()
    assert resolved.is_absolute()
    assert ".." not in str(resolved)
    assert not trial.exists()  # resolving is not creating


# --- the cache the bytes live in, and which revision they are -------------------

def blob(data: bytes) -> str:
    """A git blob hash, computed here so the tool is not checked against itself."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def hub_cache(tmp_path: Path, monkeypatch, *, revision: str = PIN, names=("model.bin", "config.json"), published=None):
    """A hub-cache-shaped tree: `snapshots/<revision>/<name>` plus, when asked, the
    `trees/<revision>.json` table `huggingface_hub` writes when it fetches."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    repository = tmp_path / ("models--" + TEACHER_REPO.replace("/", "--"))
    for name in names:
        path = repository / "snapshots" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BLOB)
    if published is not None:
        tree = repository / "trees"
        tree.mkdir(parents=True, exist_ok=True)
        (tree / f"{revision}.json").write_text(json.dumps({"files": published}), encoding="utf-8")
    return repository


def test_the_tool_and_the_loading_engine_open_the_same_cache(tmp_path, monkeypatch):
    """One root, both callers, including when the machine redirects it.

    A digest recorded against bytes the loader never opens is evidence about a file
    nothing runs: the engine and the provisioner have to agree even on a machine where
    the cache has moved.
    """
    for var in ("HF_HOME", "HF_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert hf_hub_cache_root() == tmp_path / "home" / "hub"
    assert Path(default_cache_root()) == hf_hub_cache_root()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert Path(default_cache_root()) == hf_hub_cache_root() == tmp_path / "explicit"
    assert hf_snapshot_dir(TEACHER_REPO).parent == hf_hub_cache_root()


def test_a_pinned_asset_is_read_only_from_the_snapshot_of_that_commit(tmp_path, monkeypatch):
    """The pin is checked, not quoted: the newest directory on disk is not the answer."""
    repository = hub_cache(tmp_path, monkeypatch)
    snapshots = repository / "snapshots"
    for name in ("model.bin", "config.json"):
        path = snapshots / ("0" * 40) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"different-revision-bytes")
    # Make the unpinned revision the newest, so a "latest directory" fallback would win.
    os.utime(snapshots / PIN, (1_600_000_000, 1_600_000_000))
    os.utime(snapshots / ("0" * 40), (1_700_000_000, 1_700_000_000))
    pinned = snapshot_files(TEACHER_REPO, PIN)
    assert {path.parent for path in pinned} == {snapshots / PIN}
    assert all(path.read_bytes() == BLOB for path in pinned)
    # No snapshot for an unknown commit, and nothing digested in its place.
    assert snapshot_files(TEACHER_REPO, "f" * 40) == []
    # Without a pin the newest directory wins - which is exactly the guess the pin removes.
    assert snapshot_files(TEACHER_REPO)[0].read_bytes() == b"different-revision-bytes"


def test_the_teacher_is_pinned_to_the_revision_the_repository_already_records():
    """The pin has to be in the tool, or it is a sentence in a provenance note."""
    assert CATALOGUE[TEACHER]["revision"] == PIN
    evidence = json.loads((ROOT / "artifacts" / "provenance" / "faster-whisper-large-v3.json").read_text(encoding="utf-8"))
    assert evidence["revision_pinned"] == PIN
    assert evidence["licence_spdx"] == "NOASSERTION"
    assert len(evidence["files"]) == 7
    weights = next(row for row in evidence["files"] if row["path"] == "model.bin")
    assert weights["bytes"] == 3087284237
    assert weights["sha256_computed_locally"] == weights["published_lfs_sha256"]


# --- what the source published about these bytes --------------------------------

def test_a_published_digest_that_matches_is_recorded_with_its_basis(tmp_path, monkeypatch):
    repository = hub_cache(tmp_path, monkeypatch, names=("config.json",), published={"config.json": {"size": len(BLOB), "blob_id": blob(BLOB)}})
    path = repository / "snapshots" / PIN / "config.json"
    published = published_digests(repository, PIN)
    basis = check_published_digest(path, "snapshots/x/config.json", sha256(path), path.stat().st_size, published["config.json"])
    assert "git blob" in basis
    # tree keys drop the snapshot prefix the way the hub's own index names the file.
    assert tree_key("snapshots/abc/nested/dict/jieba.dict.utf8") == "nested/dict/jieba.dict.utf8"
    assert tree_key("model.bin") == "model.bin"


def test_a_digest_mismatch_is_fatal_rather_than_a_note(tmp_path, monkeypatch):
    repository = hub_cache(tmp_path, monkeypatch, names=("model.bin",), published={"model.bin": {"size": len(BLOB), "lfs_sha256": "0" * 64}})
    path = repository / "snapshots" / PIN / "model.bin"
    with pytest.raises(ConfigurationError) as exc:
        check_published_digest(path, "snapshots/x/model.bin", sha256(path), path.stat().st_size, published_digests(repository, PIN)["model.bin"])
    assert "does not match" in str(exc.value)
    assert "model.bin" in str(exc.value)


def test_a_size_mismatch_is_fatal_too(tmp_path, monkeypatch):
    repository = hub_cache(tmp_path, monkeypatch, names=("model.bin",), published={"model.bin": {"size": len(BLOB) + 1, "blob_id": blob(BLOB)}})
    path = repository / "snapshots" / PIN / "model.bin"
    with pytest.raises(ConfigurationError) as exc:
        check_published_digest(path, "snapshots/x/model.bin", sha256(path), path.stat().st_size, published_digests(repository, PIN)["model.bin"])
    assert "size" in str(exc.value)


def test_a_pinned_asset_with_no_cached_published_record_is_refused(tmp_path, monkeypatch):
    """C2: a record this cache does not have is not a statement about upstream.

    This used to answer "no published digest" and carry on, so a hub cache that happened
    to be missing ``trees/<revision>.json`` produced a manifest claiming the source
    publishes nothing - while the fatal comparison the module header promises never ran.
    An unpinned asset still asks for nothing, because there is no revision whose record
    could be looked up, and that difference is the whole correction.
    """
    repository = hub_cache(tmp_path, monkeypatch, names=("model.bin",))
    path = repository / "snapshots" / PIN / "model.bin"
    with pytest.raises(ConfigurationError) as exc:
        published_digests(repository, PIN)
    assert "no published-digest record is cached" in str(exc.value)
    assert PIN in exc.value.details["expected"] and "trees" in exc.value.details["expected"]
    assert published_digests(repository, "") == {}
    # A file with no published entry is still reported per file, and never assumed to match.
    basis = check_published_digest(path, "snapshots/x/model.bin", sha256(path), path.stat().st_size, {})
    assert basis == "no digest published by the source for this file"
    assert git_blob_sha1(path) == blob(BLOB)


@pytest.mark.parametrize(
    "body",
    ["", "{ not json", "[]", '{"files": {}}', '{"format_version": 1}', '{"files": []}'],
    ids=["empty-file", "truncated-json", "json-list", "empty-table", "no-files-key", "files-not-a-table"],
)
def test_a_malformed_published_record_refuses_instead_of_silencing_the_check(tmp_path, monkeypatch, body):
    """Every shape of unreadable metadata, including the ones that are valid JSON.

    A syntactically legal file that lists nothing is the dangerous case: it satisfies an
    `is_file()` check, parses, and then compares no file at all - the same outcome as the
    absent record that this test's siblings cover.
    """
    repository = hub_cache(tmp_path, monkeypatch, names=("model.bin",), published={"model.bin": {"size": 1}})
    (repository / "trees" / f"{PIN}.json").write_text(body, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        published_digests(repository, PIN)


def test_a_pinned_teacher_snapshot_with_no_cached_record_provisions_nothing(tmp_path, monkeypatch):
    """The refusal at the seam that writes the file, not only inside the helper.

    `build_manifest` raising is what makes `main` print REFUSED and never reach
    `save_manifest`, so this is the difference between a soft sentence in a note and a
    manifest that does not exist.
    """
    hub_cache(tmp_path, monkeypatch, names=("model.bin", "config.json"))  # snapshot, no trees file
    files = snapshot_files(TEACHER_REPO, PIN)
    assert len(files) == 2  # the bytes are there; only the published record is not
    with pytest.raises(ConfigurationError) as exc:
        build_manifest(TEACHER, hf_snapshot_dir(TEACHER_REPO), files, provision_args("measurement"))
    assert "no published-digest record" in str(exc.value)


def test_the_digest_notice_says_what_a_match_does_not_prove():
    both = digest_notice(3, 2, "2026-10-14T00:00:00")
    assert "2/3 files matched a digest the source publishes" in both
    assert "not a licence" in both
    assert "does not prove that a converted model came from the model it claims to convert" in both
    none = digest_notice(1, 0, "2026-10-14T00:00:00")
    assert "upstream publishes no digest for this artefact" in none
    assert "pins drift-detection not authenticity" in none


def test_the_digest_notice_does_not_call_a_missing_record_a_statement_about_upstream():
    """The three ways a comparison does not happen say different things, and get different words.

    Only one of them is about the source. The other two are about this cache, and a
    manifest that described them as upstream silence would be believed in exactly the
    direction that matters: nobody re-fetches a record they think was never published.
    """
    stamp = "2026-10-14T00:00:00"
    unpinned = digest_notice(2, 0, stamp, "unpinned")
    assert "no revision pin" in unpinned
    assert "upstream publishes no digest" not in unpinned
    read = digest_notice(2, 0, stamp, "read")
    assert "names none of these files" in read
    assert "upstream publishes no digest" not in read
    # A tarball really does publish nothing, and that stays the plain statement it was.
    assert "upstream publishes no digest for this artefact" in digest_notice(2, 0, stamp, "none")


def test_a_manifest_from_a_pinned_snapshot_records_the_revision_it_verified(tmp_path, monkeypatch):
    """Version, pin, licence and the measured flag, all in the record the tool writes."""
    repository = hub_cache(
        tmp_path,
        monkeypatch,
        published={
            "model.bin": {"size": len(BLOB), "blob_id": blob(b"version https://git-lfs.github.com/spec/v1"), "lfs_sha256": hashlib.sha256(BLOB).hexdigest()},
            "config.json": {"size": len(BLOB), "blob_id": blob(BLOB)},
        },
    )
    files = snapshot_files(TEACHER_REPO, PIN)
    manifest = build_manifest(TEACHER, repository, files, provision_args("measurement"))
    assert manifest.version == PIN
    assert manifest.use_scope is AssetScope.MEASUREMENT
    assert manifest.provenance.licence_spdx == "NOASSERTION"
    assert manifest.provenance.redistributable is False
    assert manifest.requirements.measured is False  # a recorded RAM estimate is not a measurement
    assert all(asset.relative_path.startswith(f"snapshots/{PIN}/") for asset in manifest.assets)
    assert "2/2 files matched a digest the source publishes" in manifest.provenance.licence_notice
    assert f"Pinned to revision {PIN}." in manifest.notes
    assert "Recorded without product approval" in manifest.notes
    assert manifest.approved_on == ""  # --approved-by names a person; it does not date an approval
    # U4: the two figures a later reader is most likely to mistake for measurements.
    assert "Declared, not measured" in manifest.notes
    assert "the 3300 MiB RAM figure is carried as an estimate" in manifest.notes
    assert "quantization 'ct2-default' names the conversion the catalogue declares" in manifest.notes


def test_a_snapshot_that_does_not_match_the_published_digest_writes_no_manifest(tmp_path, monkeypatch):
    repository = hub_cache(tmp_path, monkeypatch, names=("model.bin",), published={"model.bin": {"size": len(BLOB), "lfs_sha256": "deadbeef" * 8}})
    files = snapshot_files(TEACHER_REPO, PIN)
    with pytest.raises(ConfigurationError) as exc:
        build_manifest(TEACHER, repository, files, provision_args("measurement"))
    assert "model.bin" in str(exc.value)


# --- who decides approval, and who decides scope --------------------------------

def test_the_teacher_is_catalogued_for_measurement_only():
    spec = CATALOGUE[TEACHER]
    assert spec["allowed_scope"] is AssetScope.MEASUREMENT
    with pytest.raises(ConfigurationError) as exc:
        check_scope_contract(TEACHER, spec, AssetScope.PRODUCT)
    assert "not provisionable" in str(exc.value)
    check_scope_contract(TEACHER, spec, AssetScope.MEASUREMENT)
    # An asset the catalogue does not lock stays open on both axes: the entry decides,
    # the caller never does.
    check_scope_contract("faster-whisper-base", CATALOGUE["faster-whisper-base"], AssetScope.PRODUCT)
    check_scope_contract("faster-whisper-base", CATALOGUE["faster-whisper-base"], AssetScope.MEASUREMENT)


def test_a_product_scope_for_the_teacher_is_refused_before_any_manifest(tmp_path):
    directory, files = make_asset_dir(tmp_path, ["model.bin"])
    with pytest.raises(ConfigurationError):
        build_manifest(TEACHER, directory, files, provision_args("product"))


def test_a_name_alone_is_not_the_licence_gate_s_central_fact(tmp_path):
    """``--approved-by`` is evidence about who authorised the recording, not approval."""
    directory, files = make_asset_dir(tmp_path, ["model.bin"])
    manifest = build_manifest("faster-whisper-base", directory, files, provision_args("product"))
    assert manifest.approved_by == "tester@example.com"
    assert manifest.review is ReviewStatus.UNREVIEWED
    assert manifest.approved_on == ""
    assert "Recorded without product approval" in manifest.notes
    assert to_dict(manifest)["review"] == "unreviewed"


def test_product_approval_is_a_separate_deliberate_act(tmp_path):
    directory, files = make_asset_dir(tmp_path, ["model.bin"])
    manifest = build_manifest("faster-whisper-base", directory, files, provision_args("product", grant_product_approval=True))
    assert manifest.review is ReviewStatus.APPROVED
    assert manifest.approved_on
    assert "Recorded without product approval" not in manifest.notes


def test_an_approval_flag_cannot_be_attached_to_a_measurement_asset():
    with pytest.raises(ConfigurationError) as exc:
        review_state(provision_args("measurement", grant_product_approval=True))
    assert "measurement" in str(exc.value)
    assert review_state(provision_args("measurement")) is ReviewStatus.UNREVIEWED
