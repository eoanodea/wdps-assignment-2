import torch

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer

from torch import Tensor
import torch.nn


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
        # def PlattScaler(model):
        #     # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
        #     bias = torch.Tensor([1,])
        #     weight = torch.Tensor([1,]) 
        #     score = model.get_scorer().score_emb(s_emb, p_emb, o_emb, combine)
        #     return 1/(1+torch.exp(-(weight * score + bias)))

        # n = len(self.models)
        # scores = (1/n) + sum([PlattScaler(model) for model in self.models])
        print(self.models[0].get_scorer().score_emb(s_emb, p_emb, o_emb, combine))
        return self.models[0].get_scorer().score_emb(s_emb, p_emb, o_emb, combine)

        # embedder().score_sp(s,p)
        # o = embedder()                =/=  rescal()
        # perform_setting(embedder)     =/=  rescal()
        #
        # -> embedder().score_emb(s,p,o,combine) ---> rescal().score_emb(s,p,o,combine) 


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

    # S_embedder = entity embedder : KgeEmbedder
    # O_embedder = entity embedder : KgeEmbedder
    # P embedder = relation embedder : KgeEmbedder

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        for model in self.models:
            s = model.get_s_embedder().embed(s)


            real_s=s


        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, combine="spo").view(-1)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)
        
        return self._scorer.score_emb(s, p, o, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
        o = self.get_o_embedder().embed(o)
        p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o, combine="_po")

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s = self.get_s_embedder().embed(s)            
        o = self.get_o_embedder().embed(o)
        if p is None:
            p = self.get_p_embedder().embed_all()
        else:
            p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o, combine="s_o")

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
            po_scores = self._scorer.score_emb(all_entities, p, o, combine="_po")
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp_")
            po_scores = self._scorer.score_emb(all_subjects, p, o, combine="_po")
        return torch.cat((sp_scores, po_scores), dim=1)

    def load(self, models):
        self.models = models
        self._scorer.load(models)