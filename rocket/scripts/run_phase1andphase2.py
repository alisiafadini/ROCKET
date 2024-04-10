import uuid
from pathlib import Path
from rocket.refinement import RocketRefinmentConfig, run_refinement

WORKING_DIRECTORY = Path("/net/cci/alisia/openfold_tests/run_openfold/test_cases")
ALL_DATASETS = ["6lzm"]

# ALL_DATASETS = [
#     "7aoj",
#     "7dms",
#     "7dnt",
#     "7dtr",
#     "7e3z",
#     "7ecd",
#     "7edc",
#     "7ejg",
#     "7eyj",
#     "7fiu",
#     "7kzh",
#     "7msl",
#     "7o51",
#     "7pgk",
#     "7pt5",
#     "7qdv",
#     "7raw",
#     "7rm7",
#     "7rpy",
#     "7rr3",
#     "7s3l",
#     "7sez",
#     "7t26",
#     "7t7y",
#     "7t85",
#     "7tbs",
#     "7tfq",
#     "7trw",
#     "7u2r",
#     "7unn",
#     "7v6p",
#     "7v9h",
#     "7vnx",
#     "7x4e",
# ]


def generate_phase1_config(
    *,
    file_root: Path,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 0,
) -> RocketRefinmentConfig:

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=WORKING_DIRECTORY,
        batch_sub_ratio=1.0,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=False,
        verbose=False,
        bias_version=3,
        # iterations=50,
        iterations=2,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=0.5,
        multiplicative_learning_rate=1.0,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=1e-11,
        b_threshold=10.0,
        note="phase 1",
    )

    return phase1_config


def generate_phase2_config(
    *,
    phase1_uuid: uuid.UUID,
    file_root: Path,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 0,
) -> RocketRefinmentConfig:

    output_directory_path = f"{WORKING_DIRECTORY}/{file_root}/outputs/{phase1_uuid}"

    starting_bias_path = Path(f"{output_directory_path}/best_msa_bias.pt")
    starting_weights_path = Path(f"{output_directory_path}/best_feat_weights.pt")

    for p in [starting_bias_path, starting_weights_path]:
        if not p.exists():
            raise IOError(f"no: {p}")

    phase2_config = RocketRefinmentConfig(
        file_root=file_root,
        path=WORKING_DIRECTORY,
        batch_sub_ratio=1.0,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=False,
        verbose=False,
        bias_version=3,
        # iterations=300,
        iterations=2,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=0.001,
        multiplicative_learning_rate=0.001,
        weight_decay=0.0001,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=1e-11,
        b_threshold=10.0,
        note="phase 2",
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
    )

    return phase2_config


def run_both_phases_single_dataset(*, file_root) -> None:

    phase1_config = generate_phase1_config(file_root=file_root)
    phase1_uuid = run_refinement(config=phase1_config)

    phase2_config = generate_phase2_config(phase1_uuid=phase1_uuid, file_root=file_root)
    phase2_uuid = run_refinement(config=phase2_config)


def run_both_phases_all_datasets() -> None:
    for file_root in ALL_DATASETS:
        run_both_phases_single_dataset(file_root=file_root)


if __name__ == "__main__":
    run_both_phases_all_datasets()
