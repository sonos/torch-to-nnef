"""Functionalize in-place writes done through views in traced JIT graphs.

`torch.jit.trace` SSA-renames reads of a tensor mutated DIRECTLY by an
in-place op (later reads point at the op's output value), but a mutation
done THROUGH A VIEW (`buf[:, :, i] = x`, traced as `slice`/`select` +
`aten::copy_`) leaves later reads of the parent buffer pointing at the
pre-write value. Executing eagerly this is invisible (memory aliasing),
but any purely functional backend such as NNEF silently loses the write.
HF transformers hits this pattern in every recurrent/linear-attention
model (`core_attn_out[:, :, i] = ...`, cache `copy_` updates, ...).

`functionalize_view_inplace_copy` rewrites each such `aten::copy_` into
the functional `aten::select_scatter` / `aten::slice_scatter` chain and
re-points every later reader of the root buffer (and of the intermediate
view values) at the updated value.

Writes that cannot be expressed this way (strided slices, view chains
through transpose/reshape/... aliases) are left untouched; when the
pre-write value is observably read afterwards, a loud warning names the
offending source location (e.g. HF `apply_interleaved_mrope`, whose
lost write is numerically a no-op for text-only position ids).
"""

import logging
import typing as T

import torch

LOGGER = logging.getLogger(__name__)

_SCATTERABLE_VIEW_KINDS = ("aten::slice", "aten::select")


def _node_order(graph) -> T.Dict[T.Any, int]:
    return {n: i for i, n in enumerate(graph.nodes())}


def _later_uses(value, node, order, fresh=frozenset()):
    """Uses of `value` topologically after `node` (graph Return included).

    `fresh` holds nodes created by the rewrite itself: they represent
    the write and must never be re-pointed at their own result.
    """
    node_pos = order[node]
    return [
        u
        for u in value.uses()
        if u.user not in fresh
        # a user absent from the order map is the graph Return node,
        # which reads the final value: always "after".
        and (u.user not in order or order[u.user] > node_pos)
    ]


def _const_int(value) -> T.Optional[int]:
    if value.node().kind() != "prim::Constant":
        return None
    ivalue = value.toIValue()
    return ivalue if isinstance(ivalue, int) else None


def _chain_is_scatterable(views) -> bool:
    for view in views:
        if view.kind() == "aten::slice" and _const_int(view.inputsAt(4)) != 1:
            return False
    return True


def _write_is_observable(node, views, root, order) -> bool:
    if _later_uses(root, node, order):
        return True
    return any(_later_uses(v.output(), node, order) for v in views)


def _rewrite_one(graph, node, views, root, src, order) -> None:
    """Emit the scatter chain for one `copy_(view_of_root, src)`."""
    fresh = set()
    # Build the functional update bottom-up: innermost view first.
    # views[0] produces the copy_ destination, views[-1] reads root.
    updated = src
    for view in views:
        parent_val = view.inputsAt(0)
        if view.kind() == "aten::select":
            new_node = graph.create(
                "aten::select_scatter",
                [parent_val, updated, view.inputsAt(1), view.inputsAt(2)],
            )
        else:  # aten::slice
            new_node = graph.create(
                "aten::slice_scatter",
                [
                    parent_val,
                    updated,
                    view.inputsAt(1),
                    view.inputsAt(2),
                    view.inputsAt(3),
                    view.inputsAt(4),
                ],
            )
        new_node.insertBefore(node)
        new_node.output().setType(parent_val.type())
        fresh.add(new_node)
        updated = new_node.output()
    new_root = updated

    # Re-point later readers of the root at the updated buffer.
    for use in _later_uses(root, node, order, fresh):
        use.user.replaceInputWith(root, new_root)

    # Later readers of intermediate view values (and of the copy_ output,
    # which is the tracer's post-write alias of the destination view) must
    # see the written data: rebuild the view chain on the updated root.
    rebuilt_parent = new_root
    rebuilt_for_dst = None
    for view in reversed(views):
        clone_inputs = [rebuilt_parent] + [
            view.inputsAt(i) for i in range(1, view.inputsSize())
        ]
        clone = graph.create(view.kind(), clone_inputs)
        clone.insertBefore(node)
        clone.output().setType(view.output().type())
        fresh.add(clone)
        for use in _later_uses(view.output(), node, order, fresh):
            use.user.replaceInputWith(view.output(), clone.output())
        rebuilt_parent = clone.output()
        rebuilt_for_dst = clone.output()

    node.output().replaceAllUsesWith(rebuilt_for_dst)
    node.destroy()


def functionalize_view_inplace_copy(graph) -> int:
    """Rewrite `copy_` through slice/select views into functional scatters.

    Returns the number of `aten::copy_` nodes rewritten. Chains that
    cannot be functionalized (strided slice, non-slice/select alias in
    the chain root) are left as-is; if their pre-write value is read
    afterwards a warning names the write's source location, since the
    exported graph then keeps the pre-write data.
    """
    n_rewritten = 0
    warned: T.Set[str] = set()
    while True:
        order = _node_order(graph)
        target = None
        for node in graph.nodes():
            if node.kind() != "aten::copy_":
                continue
            dst = node.inputsAt(0)
            views = []
            cur = dst
            while cur.node().kind() in _SCATTERABLE_VIEW_KINDS:
                views.append(cur.node())
                cur = cur.node().inputsAt(0)
            if not views:
                # Direct (non-view) destination: the tracer SSA-renames
                # later reads to this node's output, which the aten
                # `copy` handler aliases to the source. Nothing to do.
                continue
            root = cur
            if not _chain_is_scatterable(views):
                if _write_is_observable(node, views, root, order):
                    loc = str(node.sourceRange()).split("\n", 1)[0]
                    if loc not in warned:
                        warned.add(loc)
                        LOGGER.warning(
                            "in-place write through a non-functionalizable "
                            "view chain (strided slice); the exported graph "
                            "keeps the PRE-write value for later readers. "
                            "Source: %s",
                            loc,
                        )
                continue
            target = (node, views, root, node.inputsAt(1))
            break
        if target is None:
            break
        node, views, root, src = target
        _rewrite_one(graph, node, views, root, src, order)
        n_rewritten += 1

    if n_rewritten:
        torch._C._jit_pass_dce(graph)
        LOGGER.info(
            "functionalized %d in-place copy_ through views "
            "(select_scatter/slice_scatter)",
            n_rewritten,
        )
    return n_rewritten
