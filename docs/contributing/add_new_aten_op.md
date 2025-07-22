# :+1: Add new aten / prim / quantized operator

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Contribute a new operator support

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

PyTorch internal representation (IR) contains more than 10^3^ operators (and less than 10^4^).
[Aten](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html) is the name of the underling *C++* namespace in which most of the PyTorch computational operators are specified.
[Looking at the **core list**](https://docs.pytorch.org/docs/main/torch.compiler_ir.html) in the PyTorch IR, it may seems at first there is only: <200 main ops to support (not accounting quantized & prims namespaces). Sadly these external documentations are partial, in order to keep an exhaustive track of what is supported we maintain a [generated compatibility list](./supported_operators.md).

While the most common operators are already supported in `torch_to_nnef`, this list is ever expanding, so there is always a need to catch-up when a new model end up adopting one of those.

In the development of this library we add operator translation (support) on a per need basis (aka we need to export a new model, ow it misses this and that operator, let's implement it). There is no commitment by SONOS to support them all, but contribution are always welcome.

In this tutorial we will share with you how to contribute a new operator.

## Checklist

To implement a new operator we need to follow the following steps:

- [ ] 0. Ensure this operator make sense in your targeted engine (by example `copy`, device layout ect should be `nop`, implementation detail in most inference engines)
- [ ] 1. :material-test-tube: Add few unit-test covering the operator in [test_primitive.py](#)
- [ ] 2. :material-bug-check: Check we obtain as expected the following form of exception:

```bash
torch_to_nnef.exceptions.TorchToNNEFNotImplementedError:
'$operator_aten_name' operator as not yet been translated to NNEF or registred
```

- [ ] 3. :material-code-braces:  Implement a translation in Python in one of the `torch_to_nnef.op.aten`
- [ ] 4. :material-check-all: Ensure test-suite pass
- [ ] 5. :material-ruler-square: Ensure coding guideline are respected
- [ ] 6. :material-source-pull: submit the Pull request (PR)

Obviously this list is indicative as in some infortunate cases:

- The operator does not exists in targeted inference engine: please link the associated PR from this engine as reference (by example [tract](https://github.com/sonos/tract/pulls))
- There is a bug between 2. and 3.: in that case maybe you can [file an issue](./guidelines.md) or [try to debug](./debugging.md) yourself

## <span style="color:#6666aa">**:material-step-forward: Step 0.**</span> Ensure targeted inference engine compatibility

As we stated in the checklist upper, each targeted inference engine is different (today: KhronosNNEF & TractNNEF are the only included in `torch_to_nnef`).

In the case of `tract`, the engine:

- decides on which device it runs
- what is the memory layout
- when copy should happen inside the neural network operation chains.
- willingly lack operations relative to losses and back-propagation (operations containing `backward` in their names)
- does not handle today `sparse` tensors, and quantization support is partial.
- was developed first with audio and NLP usecase in-mind so there may be a significant portion of image specific operators
that are still missing (implementation is welcome in tract repository side).
- is not a general purpose linear algebra library, so most specialized operations will certainly be missing.

This set of constraint remove a whole class of operators that are used in PyTorch, if you are unsure
about the operator you are willing to support just contact directly the maintainers of this project in the [discussion section of tract](https://github.com/sonos/tract/discussions).

## <span style="color:#6666aa">**:material-step-forward: Step 1.**</span> Adding unit tests

Let's checkout the git project and create a new branch to `torch_to_nnef` named:
 `feat/{inference_target}-aten-{aten_operator_name}` where inference target is `tract`,`khronos`
and aten operator is one of [this list](./supported_operators.md), still unsupported.

After that you can edit the file named: `./tests/test_primitive.py` and at the end of it
after the last `test_suite.add`, add the following **temporary** line:

```python
test_suite.reset()
```

After that you can copy one of our `test_suite.add` observable upper with proper call
to your unsupported torch operation by example:

```python
test_suite.add(
    torch.randn(5, 3),
    UnaryPrimitive(torch.svd),
    # filter to specific inference engine to skip from test
    inference_conditions=skip_khronos_interpreter,
)
```

Side note here: tract as no reason to expose Singular Value Decomposition (this is not part of most neural network,
but you can argue in tract discussions if you feel that's a need).

After that you can run the test with the command:

```bash
py.test tests/test_primitive.py::test_primitive_export
```

If you run it as such there should be 2 failed test case. Why 2 ? because give our filter definition in our test it run on last 2 major versions of **tract**.

What if you want to focus on 1 test case ? just run:

```bash
T2N_TEST_TRACT_VERSION="0.21.13" py.test tests/test_primitive.py::test_primitive_export
```

In this case you will test on 0.21.13 version but you can set it to any version
released at the time in [tract repository](https://github.com/sonos/tract/releases).

If you are willing to test a specific custom version of tract instead you can directly
specify the path to tract cli binary it as such:

```bash
T2N_TEST_TRACT_PATH="$HOME/.../tract" py.test tests/test_primitive.py::test_primitive_export
```

Finally you may want to focus on non tract inferences and set the:

```bash
T2N_TEST_SKIP_TRACT=1 py.test tests/test_primitive.py::test_primitive_export
```

Other useful environment variable you can activate are:

- `DUMP_DIRPATH={my_dirpath}` that will dump all `.nnef.tgz` from successful tests (useful to create a zoo of use-case), warning that may be a lot
- `DEBUG=1` that will build and fill a directory `failed_tests` when you run tests. It will contains all dumps of models that are not passing test suite but still are able to be exported to NNEF (either because of a translation code error or a bug in the targeted inference engine), with ad-hoc information useful to debug.

## <span style="color:#6666aa">**:material-step-forward: Step 2.**</span> Un-Implemented check

While adding new operators test it may happen you do not observe the error (following example in step 1.)

```bash
torch_to_nnef.exceptions.TorchToNNEFNotImplementedError:
'svd' operator as not yet been translated to NNEF or registred
```

If that happen you can either [file an issue](./guidelines.md) or [try to debug](./debugging.md) yourself.

## <span style="color:#6666aa">**:material-step-forward: Step 3.**</span> Implement translation

It's now time to add your operator translation !

A lot of example exists in the `torch_to_nnef.op.aten` sub modules.
Each sub-module is organized by theme. please try to find the one that is the closest
from your operator or put it in `other` if not.

There is mostly 2 kind of operators

- Those that are directly mapping to [NNEF spec](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html)
and are 1 to 1 tensor transformation in that case just add it in the map in `torch_to_nnef.op.aten.unary`: `GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES` or `REMAP_ATEN_OP_NAMES`.

- Those that need a bit of mapping:

```python title="Example of straight mapping"
@OP_REGISTRY.register(["bitwise_or"]) # (1)!
def bitwise_or(node, op_helper, inference_target, **kwargs): # (2)!
    assert len(node.outputs) == 1
    if not isinstance(inference_target, TractNNEF): # (3)!
        raise TorchToNNEFNotImplementedError(inference_target)
    op_helper.unary_output_op_without_attr( # (4)!
        nnef_op_type="tract_core_bitor", node=node
    )
    return ["tract_core"] # (5)!
```

1. OP_REGISTRY is by convention always declared on top of module it's the registry that accumulate the translations. if you call `.register()` it will take the name of the function as reference for the operator to translate. if you provide an array like `.register(["a", "b", "c"])` all aten operators named `a`, `b` and `c` will be mapped here.
2. The complete signature of the function is evolving but as of now is: `g: nnef.Graph`, `node: torch_to_nnef.torch_graph.TorchOp`, `name_to_tensor: T.Dict[str, nnef.tensor]`, `null_ref: nnef.tensor`, `torch_graph: torch_to_nnef.torch_graph.TorchModuleIRGraph`, `inference_target: torch_to_nnef.inference_target.base.InferenceTarget`, `aten_op_id: str`, `op_helper: torch_to_nnef.op.helper.OpHelper` obviously a lot of those parameters are often unneeded hence the `**kwargs`. Basically our goal is always to translate what is in `node` the best we can in `g` while keeping `name_to_tensor` up-to-date. `OpHelper` is a 'newly' introduced builder to simplify creation of classic translation pattern.
3. Often you may want to support only for specific `inference_target` Type or Version this is an concrete example of how this can look like
4. Here we use the helper to declare a new operator that will have a single output from a single input named in NNEF graph `tract_core_bitor`
5. By default translation function can return None or empty array but if an array of string is provided, it will automatically try to load the associated fragment in [`torch_to_nnef.op.fragment`](/reference/torch_to_nnef/op/fragment/#torch_to_nnef.op.fragment.Fragment)

Here we added tooltips on each part to explains the best we could.

## <span style="color:#6666aa">**:material-step-forward: Step 4.**</span> Test suite pass

At this stage you can relaunch test suite as described upper and it should pass.

Do not forget to remove the `test_suite.reset()` and relaunch the full test suite coverage.

```bash title="global cmd"
py.test
```

To ensure nothing break due to that new addition.

## <span style="color:#6666aa">**:material-step-forward: Step 5.**</span> Check written code

There is 3 things to do now run :

- [ruff](https://docs.astral.sh/ruff/) for formatting

```bash
ruff format
```

- [ruff](https://docs.astral.sh/ruff/) check for first lint

```bash
ruff check
```

- [prospector](https://pypi.org/project/prospector/) deeper more complete analysis

In term of naming convention try to follow [google style](https://google.github.io/styleguide/pyguide.html).
Please conform to those.

## <span style="color:#6666aa">**:material-step-forward: Step 6.**</span> submit the Pull request (PR)

You are now ready to create a pull request on our repo, please
note that after that all your code will be [MIT/Apache 2] licensed.
Also please follow the same prefix naming convention for your PR name and commits than for the branch.

!!! success end "Congratulation"

    :tada: you made it !
    Congratulation and thank you so much for your contribution
