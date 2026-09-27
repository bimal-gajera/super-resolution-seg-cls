"""Minimal stand-in for c2i/src/utils.py.

The real file imports lightning, wandb and psutil, none of which inference
needs. diffusion.py uses exactly one helper from it, reproduced verbatim here.
"""


def no_grad(net):
    assert net is not None, "net is None"
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    return net
