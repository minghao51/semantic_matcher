from semanticmatcher.config import Config


def test_config_loads_default_and_nested_access(tmp_path, monkeypatch):
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "default_model: base-model\ntraining:\n  num_epochs: 4\n  batch_size: 16\n"
    )

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [default_path],
    )

    cfg = Config()

    assert cfg.default_model == "base-model"
    assert cfg.training.num_epochs == 4
    assert cfg.get("training.batch_size") == 16
    assert cfg.get("missing.key", "fallback") == "fallback"


def test_config_merges_custom_overrides(tmp_path, monkeypatch):
    default_path = tmp_path / "default.yaml"
    custom_path = tmp_path / "custom.yaml"
    default_path.write_text("training:\n  num_epochs: 4\n  batch_size: 16\n")
    custom_path.write_text("training:\n  batch_size: 8\nembedding:\n  threshold: 0.9\n")

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [default_path],
    )

    cfg = Config(custom_path=custom_path)

    assert cfg.training.num_epochs == 4
    assert cfg.training.batch_size == 8
    assert cfg.embedding.threshold == 0.9


def test_config_instances_do_not_share_state(tmp_path, monkeypatch):
    first_default = tmp_path / "one.yaml"
    second_default = tmp_path / "two.yaml"
    first_default.write_text("default_model: first\n")
    second_default.write_text("default_model: second\n")

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [first_default],
    )
    first = Config()

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [second_default],
    )
    second = Config()

    assert first.default_model == "first"
    assert second.default_model == "second"


def test_config_can_load_packaged_default_when_local_sources_absent(monkeypatch):
    monkeypatch.setattr(Config, "_find_repo_root_config", lambda self: None)
    monkeypatch.setattr(Config, "_cwd_config", lambda self: None)

    package_resource = Config()._package_default_config()
    assert package_resource is not None

    cfg = Config()

    assert isinstance(cfg.to_dict(), dict)
