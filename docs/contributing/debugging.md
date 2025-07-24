# Debugging, tips & tricks

!!! info
    This section intent to be an 'unordered' collection of debugging tricks
    for **torch_to_nnef** internals.

## <span style="color:#6666aa">**:material-math-log:**</span> Logging

Setting the log level to debug in `torch_to_nnef` may help you figure out where
things broke in the first place. There is a [`torch_to_nnef.log`](/reference/torch_to_nnef/log/) exactly for that.

## <span style="color:#6666aa">**:material-bug:**</span> Errors

In `torch_to_nnef` we try to derive all possible errors from [`torch_to_nnef.exceptions.TorchToNNEFError`](/reference/torch_to_nnef/exceptions),
so it should help to interpret why issue arise, but also control it on upper level control flow.

## <span style="color:#6666aa">**:material-graph:**</span> Graph Display

If you end up debugging the internal IR construction it is very helpful to display the representation
that are built as they may involve a lot of parameters and operators, be sure to read about [this section](/contributing/internal_design/#3-internal-ir-representation),
that may help you a lot.

## <span style="color:#6666aa">**:material-code-json:**</span> .NNEF Display

Reading plain text without syntax coloring is painful and may lead to confusion.
You can use `c++` syntax display as a good proxy in most IDE. If you are a
vim user you can also use this direct [NNEF syntax plugin](https://github.com/DreamerMind/vim-nnef).
