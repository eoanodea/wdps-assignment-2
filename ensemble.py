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
        # embs = []
        # for model in self.models:
        #     s = model.get_s_embedder().embed(s_emb)
        #     p = model.get_s_embedder().embed(p_emb)
        #     o = model.get_o_embedder().embed_all()

        #     embs.append(
        #         model._scorer.score_emb(s, p, o, combine="sp_")
        #     )
        # return embs
        return self.models[0]._scorer.score_emb(s_emb, p_emb, o_emb, combine)

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