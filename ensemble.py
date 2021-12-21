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

    def load(self, models, checkpoints):
        self.models = models
    
    def PlattScaler(score):
        # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
        bias = torch.Tensor([1,])
        weight = torch.Tensor([1,])
        return 1/(1+torch.exp(-(weight * score + bias)))

    def score_spo(self, score_spos: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_spo(s, p, o, direction)
            scores.append(score)

        n = len(self.models)
        scores = (1/n) + sum([self.PlattScaler(score) for score in scores])
    
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_sp(s, p, o)
            scores.append(score)

        n = len(self.models)
        scores = (1/n) + sum([self.PlattScaler(score) for score in scores])
    
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_po(s, p, o)
            scores.append(score)

        n = len(self.models)
        scores = (1/n) + sum([self.PlattScaler(score) for score in scores])
    
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_so(s, p, o)
            scores.append(score)

        n = len(self.models)
        scores = (1/n) + sum([self.PlattScaler(score) for score in scores])
    
        return scores

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        scores = []
        for model in self.models:
            score = model.score_sp_po(s, p, o, entity_subset)
            scores.append(score)

        n = len(self.models)
        scores = (1/n) + sum([self.PlattScaler(score) for score in scores])
    
        return scores