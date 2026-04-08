import typing as T
from dataclasses import dataclass

import torch
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.remodeler import (
    IODescriptor,
    Provider,
    RemodelPlan,
    Stage,
    SubnetSignature,
)

from torch_to_nnef_nemo.export import (
    iter_export_params_for_generic_nemo_asr_model,
)
from torch_to_nnef_nemo.inspect import (
    collect_signatures as nemo_collect_signatures,
)


@dataclass
class NemoProvider(Provider):
    """NeMo remodeler provider.

    Args:
        inference_target: Export target used for discovery (Tract by default).
        skip_preprocessor: Whether to exclude the preprocessor subnet.
        split_joint_decoder: Whether to split decoder and joint.
        float_dtype: Preferred float dtype for examples.
        only_subnets: Optional subset of subnets to consider.
    """

    inference_target: TractNNEF
    skip_preprocessor: bool = False
    split_joint_decoder: bool = False
    float_dtype: T.Optional[torch.dtype] = None
    only_subnets: T.Optional[T.Collection[str]] = None

    def discover_signatures(
        self, model: torch.nn.Module, stage: Stage
    ) -> T.List[SubnetSignature]:
        """Discover per-subnet signatures for the given stage."""
        snaps = nemo_collect_signatures(
            asr_model=model,
            inference_target=self.inference_target,
            stage=stage,
            skip_preprocessor=self.skip_preprocessor,
            split_joint_decoder=self.split_joint_decoder,
            float_dtype=self.float_dtype or torch.float32,
            only_subnets=self.only_subnets,
        )
        out: list[SubnetSignature] = []
        for ss in snaps:
            out.append(
                SubnetSignature(
                    name=ss.name,
                    stage=Stage(ss.stage.value),
                    inputs=[
                        IODescriptor(
                            name=i.name,
                            shape=list(i.shape),
                            dtype=i.dtype,
                            notes=list(i.notes or []),
                        )
                        for i in ss.inputs
                    ],
                    outputs=[
                        IODescriptor(
                            name=o.name,
                            shape=list(o.shape),
                            dtype=o.dtype,
                            notes=list(o.notes or []),
                        )
                        for o in ss.outputs
                    ],
                    symbol_axes={
                        k: {int(ax): str(sym) for ax, sym in v.items()}
                        for k, v in (ss.symbol_axes or {}).items()
                    },
                )
            )
        return out

    def apply(
        self, model: torch.nn.Module, plan: RemodelPlan
    ) -> dict[str, torch.nn.Module]:
        """Apply boundary remodel plan and return wrapped subnets.

        Uses the export pipeline to construct BoundaryAdapter-wrapped modules
        following the external IO boundary described by the plan.
        """
        wrapped: dict[str, torch.nn.Module] = {}
        for ep in iter_export_params_for_generic_nemo_asr_model(
            asr_model=model,
            inference_target=self.inference_target,
            skip_preprocessor=self.skip_preprocessor,
            split_joint_decoder=self.split_joint_decoder,
            remove_unused_inputs=True,
            float_dtype=self.float_dtype,
            only_subnets=self.only_subnets,
            axis_registry=plan.registry,
        ):
            wrapped[ep.name] = ep.model
        return wrapped
