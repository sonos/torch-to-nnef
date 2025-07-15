# 8. Custom operators

In case you want control specific `torch.nn.Module` expansion to NNEF you can
create a subclass of `torch_to_nnef.op.custom_extractors.ModuleInfoExtractor`
that is defined as such:

<div class="grid cards" markdown>
- ::: torch_to_nnef.op.custom_extractors.ModuleInfoExtractor
    handler: python
</div>

To make it work you need 4 steps:

1. sub-classing it
2. defining it's `MODULE_CLASS` attribute.
3. defining it's `convert_to_nnef`
4. ensuring that the subclass you just defined is imported in your export script

This would look something like that:

```python
from torch_to_nnef.op.custom_extractors import ModuleInfoExtractor

class MyCustomHandler(ModuleInfoExtractor):
    MODULE_CLASS = MyModuleToCustomConvert

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        **kwargs
    ):
        pass
```

You can take inspiration from our own management of RNN layers like:
<div class="grid cards" markdown>
- ::: torch_to_nnef.op.custom_extractors.LSTMExtractor
    handler: python
</div>
But ultimately this is just a chain of op's that need to be written,
inside the `g` graph.
