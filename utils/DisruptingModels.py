import torch


def DisruptingModels(model):
    fc_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

    for layer_name in fc_layers:
        # if layer_name == 'fc1' or layer_name == 'fc2':
        #     continue
        layer = getattr(model, layer_name)
        weight = layer.weight
        bias = layer.bias
        layer.weight = torch.nn.Parameter(torch.randn(weight.shape))
        layer.bias = torch.nn.Parameter(torch.randn(bias.shape))

    params = []
    for m in model.models:
        params += list(m.parameters())

    for i, p in enumerate(params):
        if i > 11:
            params[i] = torch.nn.Parameter((torch.randn(p.shape)).cuda())

    # (model.models[-1]).parameters() = params[-3:]
    model.models[-1].hb = params[-1]
    model.models[-1].vb = params[-2]
    model.models[-1].W = params[-3]
    pass