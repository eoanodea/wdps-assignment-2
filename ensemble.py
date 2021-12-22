import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer

class Ensemble(KgeModel):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RelationalScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.models = []

    def load(self, models):
        self.models = models
    
    def platt_scaler(self, score) -> Tensor:
        # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
        bias = torch.Tensor([1,])
        weight = torch.Tensor([1,])
        return 1/(1+torch.exp(-(weight * score + bias)))

    def score(self, scores) -> Tensor:
        n = len(self.models)
        return (1/n) + sum([self.platt_scaler(score) for score in scores])

    def score_spo(self, score_spos: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_spo(score_spos, p, o, direction)
            scores.append(score)

        return self.score(scores)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        return self.score([model.score_sp(s, p, o) for model in self.models])

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        return self.score([model.score_po(p, o, s) for model in self.models])

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        return self.score([model.score_so(s, o, p) for model in self.models])

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        return self.score([model.score_sp_po(s, p, o, entity_subset) for model in self.models])