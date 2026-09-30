from collections import OrderedDict

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict) -> None:
        self.shadow = OrderedDict((name, value.clone()) for name, value in state_dict.items())

