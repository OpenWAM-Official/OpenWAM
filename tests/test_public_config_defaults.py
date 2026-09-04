from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_does_not_warm_start_from_a_placeholder_by_default():
    config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    assert config["training"]["finetune_ckpt_path"] is None


def test_robocoin_trim_manifest_is_opt_in():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "dataloader" / "pretrain_data" / "robocoin.yaml").read_text(encoding="utf-8")
    )

    assert config["trim_csv"] is None
