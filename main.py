# Import base libraries
import sys
import glob, os
import math

# CLI libraries
# from tqdm import tqdm
import argparse

# Torch and KGE
import torch
import kge
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.job.eval import EvaluationJob
from kge.job.eval_entity_ranking import EntityRankingJob
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
        self.model.eval()

    def evaluate(self):
        def test_model(model):
            s = torch.Tensor([0, 2,]).long()             # subject indexes
            p = torch.Tensor([0, 1,]).long()             # relation indexes
            scores = model.score_sp(s, p)                # scores of all objects for (s,p,?)
            o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects

            print("1:", o)
            print("2:",model.dataset.entity_strings(s))       # convert indexes to mentions
            print("3:",model.dataset.relation_strings(p))
            print("4:",model.dataset.entity_strings(o))
            
        print("Model 1:")
        test_model(self.model.models[0])
        print("Yeehaw Model")
        test_model(self.model)

        # job = EvaluationJob.create(
        #     self.model.config,
        #     self.model.datasets,
        #     None,
        #     self.model
        #     )
        # job.run()
    
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
