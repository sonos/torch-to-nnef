from torch_to_nnef.remodeler import IODescriptor, Stage, SubnetSignature
from torch_to_nnef.remodeler.serialize import signatures_to_json_payload


def test_signatures_to_json_payload_basic():
    sigs = [
        SubnetSignature(
            name="s",
            stage=Stage.RAW,
            inputs=[IODescriptor("x", ["B", 10], "float32", [])],
            outputs=[IODescriptor("y", [], None, [])],
            symbol_axes={"x": {0: "B"}},
        )
    ]
    p = signatures_to_json_payload(sigs, model_label="m")
    assert p["model"] == "m"
    assert p["subnets"][0]["name"] == "s"
    assert p["subnets"][0]["stage"] == "raw"
    assert p["subnets"][0]["inputs"][0]["shape"][0] == "B"
