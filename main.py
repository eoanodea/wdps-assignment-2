# Import base libraries
import sys
import glob, os
import math

# CLI libraries
from tqdm import tqdm
import argparse

# Torch and KGE
import torch
import kge
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.job.eval import EvaluationJob
from kge.model.ensemble import Ensemble

# Main Class
class Main():
    def __init__(self, checkpoints):
        models = []
        for path in checkpoints:
            checkpoint = load_checkpoint(path)
            models.append(
                KgeModel.create_from(
                    checkpoint
                )
            )
            
        self.model = KgeModel.create(models[0].config, models[0].dataset, 'ensemble')
        self.model.load(models)

    def evaluate(self):
        s = torch.Tensor([0, 2,]).long()             # subject indexes
        p = torch.Tensor([0, 1,]).long()             # relation indexes
        scores = self.model.score_sp(s, p)           # scores of all objects for (s,p,?)
        o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects
        pass
        # eval_split = self.config.get("eval.split")
        # triples = self.dataset.split(eval_split) 
        # loader = torch.utils.data.DataLoader(
        #     triples,
        #     collate_fn=self._collate,
        #     shuffle=False,
        #     batch_size=32,
        #     num_workers=self.config.get("eval.num_workers"),
        #     pin_memory=self.config.get("eval.pin_memory"),
        # )

        # device = self.config.get("job.device")
        # for batch_number, batch_coords in enumerate(loader):
        #     batch = batch_coords[0].to(device)
        #     s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]

        #     label_coords = batch_coords[1].to(device)

        #     num_entities = self.dataset.num_entities()

        #     # create sparse labels tensor
        #     labels = kge.job.util.coord_to_sparse_tensor(
        #         len(batch), 2 * num_entities, label_coords, device, float("Inf")
        #     )
            
        #     print(self.score_emb(s, p, o))

            # for model in self.models:
            #     unique_o, unique_o_inverse = torch.unique(o, return_inverse=True)
            #     o_true_scores = torch.gather(
            #         model.score_sp(s, p, unique_o),
            #         1,
            #         unique_o_inverse.view(-1, 1),
            #     ).view(-1)
            #     unique_s, unique_s_inverse = torch.unique(s, return_inverse=True)
            #     s_true_scores = torch.gather(
            #         model.score_po(p, o, unique_s),
            #         1,
            #         unique_s_inverse.view(-1, 1),
            #     ).view(-1)
                
            #     # calculate scores in chunks to not have the complete score matrix in memory
            #     # a chunk here represents a range of entity_values to score against
            #     if self.config.get("entity_ranking.chunk_size") > -1:
            #         chunk_size = self.config.get("entity_ranking.chunk_size")
            #     else:
            #         chunk_size = self.dataset.num_entities()

            #     # process chunk by chunk
            #     for chunk_number in range(math.ceil(num_entities / chunk_size)):
            #         chunk_start = chunk_size * chunk_number
            #         chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

            #         # compute scores of chunk
            #         scores = model.score_sp_po(
            #             s, p, o, torch.arange(chunk_start, chunk_end, device=device)
            #         )
            #         scores_sp = scores[:, : chunk_end - chunk_start]
            #         scores_po = scores[:, chunk_end - chunk_start :]

    # def _collate(self, batch):
    #     "Looks up true triples for each triple in the batch"
    #     label_coords = []
    #     batch = torch.cat(batch).reshape((-1, 3))
    #     for split in self.config.get("entity_ranking.filter_splits"):
    #         split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
    #             batch,
    #             self.dataset.num_entities(),
    #             self.dataset.index(f"{split}_sp_to_o"),
    #             self.dataset.index(f"{split}_po_to_s"),
    #         )
    #         label_coords.append(split_label_coords)
    #     label_coords = torch.cat(label_coords)

    #     if "test" not in self.config.get("entity_ranking.filter_splits"):
    #         test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
    #             batch,
    #             self.dataset.num_entities(),
    #             self.dataset.index("test_sp_to_o"),
    #             self.dataset.index("test_po_to_s"),
    #         )
    #     else:
    #         test_label_coords = torch.zeros([0, 2], dtype=torch.long)

    #     return batch, label_coords, test_label_coords

# Execute main functionality
if __name__ == '__main__':
    def to_list(arg):
        return [str(i) for i in arg.split(",")]

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("folders", help="The configuration folder path(s)", type=to_list)
    args = parser.parse_args()

    checkpoints = []
    for folder in args.folders:
        checkpoints.append(folder + '/checkpoint_best.pt')

    program = Main(checkpoints)
    program.evaluate()
