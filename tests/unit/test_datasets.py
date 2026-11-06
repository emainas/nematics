from nematics.datasets import dataset_names, get_dataset, get_project_paths


def test_dataset_registry_contains_all_source_systems():
    assert dataset_names() == (
        "amoeba-4ocb",
        "amoeba-5cb",
        "amoeba-8cb",
        "amoeba-cb7cb",
        "gaff-5cb",
    )


def test_dataset_paths_and_run_metadata():
    paths = get_project_paths()
    gaff = get_dataset("gaff-5cb")
    amoeba = get_dataset("amoeba-5cb")

    assert gaff.raw_path() == paths.raw_root / gaff.raw_subdir
    assert gaff.processed_path() == paths.processed_root / "5cb/gaff-zalign"
    assert gaff.temperatures_inferred
    assert [(run.id, run.temperature_k) for run in gaff.runs] == [
        ("gro", 356.0),
        ("gro2", 366.0),
        ("gro3", 376.0),
        ("gro4", 386.0),
    ]

    run_302 = next(run for run in amoeba.runs if run.temperature_k == 302.0)
    assert run_302.id == "024a_d-md-302"
    assert run_302.historical_processed_id == "003f_d-md-302"


def test_snapshot_patterns_are_checked_as_globs(tmp_path):
    gaff = get_dataset("gaff-5cb")
    for run in gaff.runs:
        run_dir = tmp_path / run.id
        run_dir.mkdir()
        (run_dir / "frame0.gro").touch()

    configured = gaff.run_trajectories(tmp_path)

    assert configured[0][1] == tmp_path / "gro" / "frame*.gro"
