import math

import torch

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer

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
        def PlattScaler(score):
            # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
            w0 = 0 # ??
            w1 = 0 # ??
            1/(1+math.exp(-(w1 * score + w0)))

        n = 1 # length of what??
        score = (1/n) + sum([PlattScaler(model._scorer.score_emb(s_emb, p_emb, o_emb, combine)) for model in self.models])

        return score

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