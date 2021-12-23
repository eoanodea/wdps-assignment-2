import torch
from torch import Tensor

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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
        self.ensemble = []

    def load(self, models):
        self.ensemble = [{model: model, scaler: None} for model in models]
        self.train_platt_scaler()
    
    def train_platt_scaler(self):
        for i, current in enumerate(self.ensemble):
            # read train/test sets
            test_set = pd.read_csv(current.model.dataset.folder + '/test.del', on_bad_lines='skip', sep = '\t', header = None, names = ['s', 'p', 'o'])
            train_set = pd.read_csv(current.model.dataset.folder + '/train.del', on_bad_lines='skip', sep = '\t', header = None, names = ['s','p','o'])
            
            train_set = train_set.iloc[:200]

            # test predictions
            test_s = torch.Tensor(test_set['s'].values).long()
            test_p = torch.Tensor(test_set['p'].values).long()         
            test_predictions = current.model.score_sp(test_s, test_p)

            # train predictions
            train_s = torch.Tensor(train_set['s'].values).long()
            train_p = torch.Tensor(train_set['p'].values).long()         
            train_o = torch.Tensor(train_set['o'].values).long()      
            train_predictions = current.model.score_sp(train_s, train_p)

            theta, index = torch.max(train_predictions, dim=-1)
        
            # real/estimated object
            real_o = current.model.dataset.entity_strings(train_o)
            estimated_o = current.model.dataset.entity_strings(index)

            # prepare X, y
            X = theta.long().numpy().reshape(-1, 1)
            y = [i == j for i, j in zip(estimated_o, real_o)]

            # train and fit model
            current[i].scaler = LogisticRegression(random_state=0)
            current[i].scaler.fit(X, y)
            
    def platt_scaler(self, i, score) -> Tensor:
        # The scalars ωm1 and ωm0 in Equation 5 denote the learned weight and bias of the logistic regression (Platt-Scaler) for the model m.
        bias = self.ensemble[i].scaler.coef_[0]
        weight = self.ensemble[i].scaler.intercept_[0]
        return 1/(1+torch.exp(-(weight * score + bias)))


    def score(self, scores) -> Tensor:
        n = len(self.ensemble)
        return (1/n) + sum([self.platt_scaler(i, score) for i, score in enumerate(scores)])

    def score_spo(self, score_spos: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        return self.score([current.model.score_spo(score_spos, p, o, direction) for current in self.ensemble])

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        return self.score([current.model.score_sp(s, p, o) for current in self.ensemble])

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        return self.score([current.model.score_po(p, o, s) for current in self.ensemble])

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        return self.score([current.model.score_so(s, o, p) for current in self.ensemble])

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        return self.score([current.model.score_sp_po(s, p, o, entity_subset) for current in self.models])