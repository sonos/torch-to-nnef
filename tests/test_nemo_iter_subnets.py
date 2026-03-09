import contextlib
from typing import Iterator

import torch


def test_iter_nemo_model_subnets_accepts_list_names(monkeypatch):
    # Import the module under test
    import torch_to_nnef.nemo_tract.export as ne

    # Fake subnet and model minimal interfaces
    class FakeSubnet:
        input_names = ["x"]
        output_names = ["y"]

    class FakeModel:
        def list_export_subnets(self):
            return ["ignored"]

        def get_export_subnet(self, name: str):
            # Ensure we do receive a string here (regression guard)
            assert isinstance(name, str)
            return FakeSubnet()

    # Patch picker to return a plain list of names
    def fake_pick_names(_model):
        return ["encoder"]

    # Patch exportable_nemo_net to avoid depending on NeMo at runtime
    @contextlib.contextmanager
    def fake_exportable_nemo_net(
        subnet_name, subnet, input_example, batch_size=1, float_dtype=None, **kw
    ) -> Iterator[object]:
        class Ctx:
            pass

        ctx = Ctx()
        ctx.input_example = [torch.zeros(1)]  # matches len(input_names)
        ctx.dynamic_axes = {}
        ctx.output_example = ()
        yield ctx

    monkeypatch.setattr(ne, "_pick_subnets_names", fake_pick_names)
    monkeypatch.setattr(ne, "exportable_nemo_net", fake_exportable_nemo_net)

    results = list(ne.iter_nemo_model_subnets(FakeModel(), batch_size=1))

    # We should receive one tuple starting with the subnet name 'encoder'
    assert results, "No subnets yielded"
    assert results[0][0] == "encoder"
