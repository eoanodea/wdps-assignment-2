import torch

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer

from kge.model.complex import ComplExScorer

class EnsembleScorer(RelationalScorer):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.models = []

    def score_emb(
        self,
        s_emb: torch.Tensor,
        p_emb: torch.Tensor,
        o_emb: torch.Tensor,
        combine: str,
    ):
        def PlattScaler(model):
            # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
            bias = torch.Tensor([1,])
            weight = torch.Tensor([1,]) 
            score = model._scorer.score_emb(s_emb, p_emb, o_emb, combine)
            return 1/(1+torch.exp(-(weight * score + bias)))

        n = len(self.models)
        scores = (1/n) + sum([PlattScaler(model) for model in self.models])

        return scores

    def load(self, models):
        self.models = models

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
            scorer=EnsembleScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.models = []

    def load(self, models):
        self.models = models
        self._scorer.load(models)