from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import TaskType


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    family: str
    task_types: tuple[TaskType, ...]
    training_modes: tuple[str, ...]
    native_multivariate: bool
    adaptation_modes: tuple[str, ...]
    optional_dependency: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["task_types"] = [task.value for task in self.task_types]
        return payload


_DEFAULT_CATALOG = (
    ModelDescriptor(
        name="naive",
        family="baseline",
        task_types=(TaskType.FORECASTING,),
        training_modes=("frozen",),
        native_multivariate=False,
        adaptation_modes=("zero_shot",),
        notes="Persistence baseline used as a deployment sanity check.",
    ),
    ModelDescriptor(
        name="patchtst",
        family="supervised_transformer",
        task_types=(TaskType.FORECASTING,),
        training_modes=("supervised",),
        native_multivariate=True,
        adaptation_modes=("source_train", "few_shot_finetune"),
        notes="Compact in-repository supervised baseline.",
    ),
    ModelDescriptor(
        name="timesfm",
        family="tsfm",
        task_types=(TaskType.FORECASTING,),
        training_modes=("frozen",),
        native_multivariate=False,
        adaptation_modes=("zero_shot", "target_scaling", "target_residual_calibration"),
        optional_dependency="tsfm",
        notes="Official TimesFM 2.5 checkpoint adapter; current industrial path is channel-wise.",
    ),
    ModelDescriptor(
        name="chronos",
        family="tsfm",
        task_types=(TaskType.FORECASTING,),
        training_modes=("frozen",),
        native_multivariate=False,
        adaptation_modes=("zero_shot", "target_scaling", "target_residual_calibration"),
        optional_dependency="tsfm",
        notes="Original Chronos adapter; current industrial path is channel-wise.",
    ),
    ModelDescriptor(
        name="chronos2",
        family="tsfm",
        task_types=(TaskType.FORECASTING,),
        training_modes=("frozen",),
        native_multivariate=True,
        adaptation_modes=("zero_shot", "target_scaling", "target_residual_calibration"),
        optional_dependency="tsfm",
        notes="Native multivariate Chronos-2 adapter using the official forecast interface.",
    ),
    ModelDescriptor(
        name="moirai2",
        family="tsfm",
        task_types=(TaskType.FORECASTING,),
        training_modes=("frozen",),
        native_multivariate=True,
        adaptation_modes=("zero_shot", "target_scaling", "target_residual_calibration"),
        optional_dependency="moirai",
        notes="Native multivariate Moirai-2 adapter through Uni2TS.",
    ),
    ModelDescriptor(
        name="timesfm_transformers",
        family="tsfm_peft",
        task_types=(TaskType.FORECASTING,),
        training_modes=("peft",),
        native_multivariate=False,
        adaptation_modes=("source_peft", "target_peft"),
        optional_dependency="peft",
        notes="Transformers TimesFM path with LoRA/PEFT support.",
    ),
    ModelDescriptor(
        name="pca_spe_product",
        family="classical_anomaly",
        task_types=(TaskType.ANOMALY,),
        training_modes=("train_prefix",),
        native_multivariate=True,
        adaptation_modes=("train_prefix_fit", "train_quantile_threshold"),
        notes=(
            "Deployment-safe PCA squared prediction error baseline with train-only scaling, "
            "thresholding, and sensor contribution ranking."
        ),
    ),
)


def model_catalog() -> list[dict[str, object]]:
    """Return product-facing metadata only for model adapters implemented in this repo."""

    return [descriptor.to_dict() for descriptor in _DEFAULT_CATALOG]


def get_model_descriptor(name: str) -> ModelDescriptor:
    for descriptor in _DEFAULT_CATALOG:
        if descriptor.name == name:
            return descriptor
    raise KeyError(f"unknown model descriptor: {name}")
