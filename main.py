import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

# download link for this checkpoint given under results above
checkpoint = load_checkpoint('/kge/local/experiments/main/checkpoint_best.pt')
model = KgeModel.create_from(checkpoint)

s = torch.Tensor([0, 2,]).long()             # subject indexes
p = torch.Tensor([0, 1,]).long()             # relation indexes
scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

print(o)
print(model.dataset.entity_strings(s))       # convert indexes to mentions
print(model.dataset.relation_strings(p))
print(model.dataset.entity_strings(o))