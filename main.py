# Import base libraries
import sys
import glob, os
import math

# CLI libraries
from tqdm import tqdm
import argparse

# Torch and KGE
import torch
from torch import Tensor

import kge
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.job.eval import EvaluationJob
from kge.job.eval_entity_ranking import EntityRankingJob
from kge.model.ensemble import Ensemble

# from collections import defaultdict

# def __initialize_hist(hists, key, job):
#     """If there is no histogram with given `key` in `hists`, add an empty one."""
#     if key not in hists:
#         hists[key] = torch.zeros(
#             [job.model.dataset.num_entities()],
#             device=job.model.config.get("job.device"),
#             dtype=torch.float,
#         )

# def hist_all(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
#     """Create histogram of all subject/object ranks (key: "all").

#     `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
#     `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
#     rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

#     """
#     __initialize_hist(hists, "all", job)
#     if job.model.config.get("entity_ranking.metrics_per.head_and_tail"):
#         __initialize_hist(hists, "head", job)
#         __initialize_hist(hists, "tail", job)
#         hist_head = hists["head"]
#         hist_tail = hists["tail"]

#     hist = hists["all"]
#     o_ranks_unique, o_ranks_count = torch.unique(o_ranks, return_counts=True)
#     s_ranks_unique, s_ranks_count = torch.unique(s_ranks, return_counts=True)
#     hist.index_add_(0, o_ranks_unique, o_ranks_count.float())
#     hist.index_add_(0, s_ranks_unique, s_ranks_count.float())
#     if job.model.config.get("entity_ranking.metrics_per.head_and_tail"):
#         hist_tail.index_add_(0, o_ranks_unique, o_ranks_count.float())
#         hist_head.index_add_(0, s_ranks_unique, s_ranks_count.float())

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
            
        # are we using KgeModel.create correctly here?
        self.model = KgeModel.create(models[0].config, models[0].dataset, 'ensemble')
        self.model.load(models, checkpoints)
        self.model.eval()

    def evaluate(self):
        pass

        job = EvaluationJob.create(
            self.model.config,
            self.model.dataset,
            None,
            #self.model.models[0]
            self.model
            )
        job.run()

        # print("YEEHAW METRIC:")
        # self._evaluate()

#     def _evaluate(self):
#         num_entities = self.model.dataset.num_entities()

#         # dictionary that maps entry of rankings to a sparse tensor containing the
#         # true labels for this option
#         labels_for_ranking = defaultdict(lambda: None)

#         # we also filter with test data if requested
#         filter_with_test = "test" not in self.model.config.get("entity_ranking.filter_splits") and self.model.config.get("entity_ranking.filter_with_test")

#         # which rankings to compute (DO NOT REORDER; code assumes the order given here)
#         rankings = (
#             ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
#         )

#         # Initiliaze dictionaries that hold the overall histogram of ranks of true
#         # answers. These histograms are used to compute relevant metrics. The dictionary
#         # entry with key 'all' collects the overall statistics and is the default.
#         hists = dict()
#         hists_filt = dict()
#         hists_filt_test = dict()

#         triples = self.model.dataset.split(self.model.config.get("eval.split"))

#         loader = torch.utils.data.DataLoader(
#             triples,
#             collate_fn=self._collate,
#             shuffle=False,
#             batch_size=self.model.config.get("eval.batch_size"),
#             num_workers=self.model.config.get("eval.num_workers"),
#             pin_memory=self.model.config.get("eval.pin_memory"),
#         )

#         current_trace = dict()
#         for batch_number, batch_coords in enumerate(loader):
#             # create initial batch trace (yet incomplete)
#             current_trace["batch"] = dict(
#                 type="entity_ranking",
#                 scope="batch",
#                 split=self.model.config.get("eval.split"),
#                 filter_splits=self.model.config.get("entity_ranking.filter_splits"),
#                 epoch=1,
#                 batch=batch_number,
#                 size=len(batch_coords[0]),
#                 batches=len(loader),
#             )

#             # construct a sparse label tensor of shape batch_size x 2*num_entities
#             # entries are either 0 (false) or infinity (true)
#             # TODO add timing information
#             batch = batch_coords[0].to(self.model.config.get("job.device"))
#             s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
#             label_coords = batch_coords[1].to(self.model.config.get("job.device"))

#             # create sparse labels tensor
#             labels = kge.job.util.coord_to_sparse_tensor(
#                 len(batch), 2 * num_entities, label_coords, self.model.config.get("job.device"), float("Inf")
#             )
#             labels_for_ranking["_filt"] = labels

#             # compute true scores beforehand, since we can't get them from a chunked
#             # score table
#             # o_true_scores = self.model.score_spo(s, p, o, "o").view(-1)
#             # s_true_scores = self.model.score_spo(s, p, o, "s").view(-1)
#             # scoring with spo vs sp and po can lead to slight differences for ties
#             # due to floating point issues.
#             # We use score_sp and score_po to stay consistent with scoring used for
#             # further evaluation.
#             unique_o, unique_o_inverse = torch.unique(o, return_inverse=True)
#             o_true_scores = torch.gather(
#                 self.model.score_sp(s, p, unique_o),
#                 1,
#                 unique_o_inverse.view(-1, 1),
#             ).view(-1)
#             unique_s, unique_s_inverse = torch.unique(s, return_inverse=True)
#             s_true_scores = torch.gather(
#                 self.model.score_po(p, o, unique_s),
#                 1,
#                 unique_s_inverse.view(-1, 1),
#             ).view(-1)

#             # default dictionary storing rank and num_ties for each key in rankings
#             # as list of len 2: [rank, num_ties]
#             ranks_and_ties_for_ranking = defaultdict(
#                 lambda: [
#                     torch.zeros(s.size(0), dtype=torch.long, device=self.model.config.get("job.device")),
#                     torch.zeros(s.size(0), dtype=torch.long, device=self.model.config.get("job.device")),
#                 ]
#             )

#             # calculate scores in chunks to not have the complete score matrix in memory
#             # a chunk here represents a range of entity_values to score against
#             if self.model.config.get("entity_ranking.chunk_size") > -1:
#                 chunk_size = self.model.config.get("entity_ranking.chunk_size")
#             else:
#                 chunk_size = self.model.dataset.num_entities()

#             # process chunk by chunk
#             for chunk_number in range(math.ceil(num_entities / chunk_size)):
#                 chunk_start = chunk_size * chunk_number
#                 chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

#                 # compute scores of chunk
#                 scores = self.model.score_sp_po(
#                     s, p, o, torch.arange(chunk_start, chunk_end, device=self.model.config.get("job.device"))
#                 )
#                 scores_sp = scores[:, : chunk_end - chunk_start]
#                 scores_po = scores[:, chunk_end - chunk_start :]

#                 # replace the precomputed true_scores with the ones occurring in the
#                 # scores matrix to avoid floating point issues
#                 s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
#                 o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
#                 o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
#                 s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()

#                 # check that scoring is consistent up to configured tolerance
#                 # if this is not the case, evaluation metrics may be artificially inflated
#                 close_check = torch.allclose(
#                     scores_sp[o_in_chunk_mask, o_in_chunk],
#                     o_true_scores[o_in_chunk_mask],
#                     rtol=float(self.model.config.get("entity_ranking.tie_handling.rtol")),
#                     atol=float(self.model.config.get("entity_ranking.tie_handling.atol")),
#                 )
#                 close_check &= torch.allclose(
#                     scores_po[s_in_chunk_mask, s_in_chunk],
#                     s_true_scores[s_in_chunk_mask],
#                     rtol=float(self.model.config.get("entity_ranking.tie_handling.rtol")),
#                     atol=float(self.model.config.get("entity_ranking.tie_handling.atol")),
#                 )

#                 if not close_check:
#                     diff_a = torch.abs(
#                         scores_sp[o_in_chunk_mask, o_in_chunk]
#                         - o_true_scores[o_in_chunk_mask]
#                     )
#                     diff_b = torch.abs(
#                         scores_po[s_in_chunk_mask, s_in_chunk]
#                         - s_true_scores[s_in_chunk_mask]
#                     )
#                     diff_all = torch.cat((diff_a, diff_b))

#                 # now compute the rankings (assumes order: None, _filt, _filt_test)
#                 for ranking in rankings:
#                     if labels_for_ranking[ranking] is None:
#                         labels_chunk = None
#                     else:
#                         # densify the needed part of the sparse labels tensor
#                         labels_chunk = self._densify_chunk_of_labels(
#                             labels_for_ranking[ranking], chunk_start, chunk_end
#                         )

#                         # remove current example from labels
#                         labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
#                         labels_chunk[
#                             s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
#                         ] = 0

#                     # compute partial ranking and filter the scores (sets scores of true
#                     # labels to infinity)
#                     (
#                         s_rank_chunk,
#                         s_num_ties_chunk,
#                         o_rank_chunk,
#                         o_num_ties_chunk,
#                         scores_sp_filt,
#                         scores_po_filt,
#                     ) = self._filter_and_rank(
#                         scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
#                     )

#                     # from now on, use filtered scores
#                     scores_sp = scores_sp_filt
#                     scores_po = scores_po_filt

#                     # update rankings
#                     ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
#                     ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
#                     ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
#                     ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

#                 # we are done with the chunk

#             # We are done with all chunks; calculate final ranks from counts
#             s_ranks = self._get_ranks(
#                 ranks_and_ties_for_ranking["s_raw"][0],
#                 ranks_and_ties_for_ranking["s_raw"][1],
#             )
#             o_ranks = self._get_ranks(
#                 ranks_and_ties_for_ranking["o_raw"][0],
#                 ranks_and_ties_for_ranking["o_raw"][1],
#             )
#             s_ranks_filt = self._get_ranks(
#                 ranks_and_ties_for_ranking["s_filt"][0],
#                 ranks_and_ties_for_ranking["s_filt"][1],
#             )
#             o_ranks_filt = self._get_ranks(
#                 ranks_and_ties_for_ranking["o_filt"][0],
#                 ranks_and_ties_for_ranking["o_filt"][1],
#             )

#             hist_hooks = [hist_all]
#             if self.model.config.get("entity_ranking.metrics_per.relation_type"):
#                 hist_hooks.append(hist_per_relation_type)
#             if self.model.config.get("entity_ranking.metrics_per.argument_frequency"):
#                 hist_hooks.append(hist_per_frequency_percentile)

#             # Update the histograms of of raw ranks and filtered ranks
#             batch_hists = dict()
#             batch_hists_filt = dict()
#             for f in hist_hooks:
#                 f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
#                 f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

#             # and the same for filtered_with_test ranks
#             if filter_with_test:
#                 batch_hists_filt_test = dict()
#                 s_ranks_filt_test = self._get_ranks(
#                     ranks_and_ties_for_ranking["s_filt_test"][0],
#                     ranks_and_ties_for_ranking["s_filt_test"][1],
#                 )
#                 o_ranks_filt_test = self._get_ranks(
#                     ranks_and_ties_for_ranking["o_filt_test"][0],
#                     ranks_and_ties_for_ranking["o_filt_test"][1],
#                 )
#                 for f in hist_hooks:
#                     f(
#                         batch_hists_filt_test,
#                         s,
#                         p,
#                         o,
#                         s_ranks_filt_test,
#                         o_ranks_filt_test,
#                         job=self,
#                     )

#             # Compute the batch metrics for the full histogram (key "all")
#             metrics = self._compute_metrics(batch_hists["all"])
#             metrics.update(
#                 self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
#             )
#             if filter_with_test:
#                 metrics.update(
#                     self._compute_metrics(
#                         batch_hists_filt_test["all"], suffix="_filtered_with_test"
#                     )
#                 )

#             max_k = min(
#                 self.model.dataset.num_entities(),
#                 max(self.model.config.get("entity_ranking.hits_at_k_s")),
#             )
            
#             hits_at_k_s = list(
#                         filter(lambda x: x <= max_k, self.model.config.get("entity_ranking.hits_at_k_s"))
#                     )

#             print(
#                 "batches: {} \n" \
#                 "mean_reciprocal_rank: {} \n" \
#                 "mean_reciprocal_rank_filtered: {} \n" \
#                 "hits_at_1: {} \n" \
#                 "hits_at_1_filtered: {} \n" \
#                 "hits_at_k_s[-1]: {} \n" \
#                 "hits_at_k_s: {} \n"\
#                 "hits_at_k_filtered: {}" \
#                 .format(
#                     str(1 + int(math.ceil(math.log10(len(loader))))),
#                     metrics["mean_reciprocal_rank"],
#                     metrics["mean_reciprocal_rank_filtered"],
#                     metrics["hits_at_1"],
#                     metrics["hits_at_1_filtered"],
#                     hits_at_k_s[-1],
#                     metrics["hits_at_{}".format(hits_at_k_s[-1])],
#                     metrics["hits_at_{}_filtered".format(hits_at_k_s[-1])],
#                 ) 
#             )


#     def _collate(self, batch):
#         "Looks up true triples for each triple in the batch"
#         label_coords = []
#         batch = torch.cat(batch).reshape((-1, 3))
#         for split in self.model.config.get("entity_ranking.filter_splits"):
#             split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
#                 batch,
#                 self.model.dataset.num_entities(),
#                 self.model.dataset.index(f"{split}_sp_to_o"),
#                 self.model.dataset.index(f"{split}_po_to_s"),
#             )
#             label_coords.append(split_label_coords)
#         label_coords = torch.cat(label_coords)

#         if "test" not in self.model.config.get("entity_ranking.filter_splits") and self.model.config.get("entity_ranking.filter_with_test"):
#             test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
#                 batch,
#                 self.model.dataset.num_entities(),
#                 self.model.dataset.index("test_sp_to_o"),
#                 self.model.dataset.index("test_po_to_s"),
#             )
#         else:
#             test_label_coords = torch.zeros([0, 2], dtype=torch.long)

#         return batch, label_coords, test_label_coords

#     def _densify_chunk_of_labels(
#         self, labels: torch.Tensor, chunk_start: int, chunk_end: int
#     ) -> torch.Tensor:
#         """Creates a dense chunk of a sparse label tensor.

#         A chunk here is a range of entity values with 'chunk_start' being the lower
#         bound and 'chunk_end' the upper bound.

#         The resulting tensor contains the labels for the sp chunk and the po chunk.

#         :param labels: sparse tensor containing the labels corresponding to the batch
#         for sp and po

#         :param chunk_start: int start index of the chunk

#         :param chunk_end: int end index of the chunk

#         :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
#         the po chunk.

#         """
#         num_entities = self.model.dataset.num_entities()
#         indices = labels._indices()
#         mask_sp = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
#         mask_po = ((chunk_start + num_entities) <= indices[1, :]) & (
#             indices[1, :] < (chunk_end + num_entities)
#         )
#         indices_sp_chunk = indices[:, mask_sp]
#         indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - chunk_start
#         indices_po_chunk = indices[:, mask_po]
#         indices_po_chunk[1, :] = (
#             indices_po_chunk[1, :] - num_entities - chunk_start * 2 + chunk_end
#         )
#         indices_chunk = torch.cat((indices_sp_chunk, indices_po_chunk), dim=1)
#         dense_labels = torch.sparse.LongTensor(
#             indices_chunk,
#             # since all sparse label tensors have the same value we could also
#             # create a new tensor here without indexing with:
#             # torch.full([indices_chunk.shape[1]], float("inf"), device=self.device)
#             labels._values()[mask_sp | mask_po],
#             torch.Size([labels.size()[0], (chunk_end - chunk_start) * 2]),
#         ).to_dense()
#         return dense_labels

#     def _filter_and_rank(
#         self,
#         scores_sp: torch.Tensor,
#         scores_po: torch.Tensor,
#         labels: torch.Tensor,
#         o_true_scores: torch.Tensor,
#         s_true_scores: torch.Tensor,
#     ):
#         """Filters the current examples with the given labels and returns counts rank and
# num_ties for each true score.

#         :param scores_sp: batch_size x chunk_size tensor of scores

#         :param scores_po: batch_size x chunk_size tensor of scores

#         :param labels: batch_size x 2*chunk_size tensor of scores

#         :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
#         objects in batch

#         :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
#         subjects in batch

#         :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
#         scores_sp and scores_po

#         """
#         chunk_size = scores_sp.shape[1]
#         if labels is not None:
#             # remove current example from labels
#             labels_sp = labels[:, :chunk_size]
#             labels_po = labels[:, chunk_size:]
#             scores_sp = scores_sp - labels_sp
#             scores_po = scores_po - labels_po
#         o_rank, o_num_ties = self._get_ranks_and_num_ties(scores_sp, o_true_scores)
#         s_rank, s_num_ties = self._get_ranks_and_num_ties(scores_po, s_true_scores)
#         return s_rank, s_num_ties, o_rank, o_num_ties, scores_sp, scores_po

#     def _get_ranks_and_num_ties(
#         self, scores: torch.Tensor, true_scores: torch.Tensor
#     ) -> (torch.Tensor, torch.Tensor):
#         """Returns rank and number of ties of each true score in scores.

#         :param scores: batch_size x entities tensor of scores

#         :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

#         :return: batch_size x 1 tensors rank and num_ties
#         """
#         # process NaN values
#         scores = scores.clone()
#         scores[torch.isnan(scores)] = float("-Inf")
#         true_scores = true_scores.clone()
#         true_scores[torch.isnan(true_scores)] = float("-Inf")

#         # Determine how many scores are greater than / equal to each true answer (in its
#         # corresponding row of scores)
#         is_close = torch.isclose(
#             scores, true_scores.view(-1, 1), rtol=float(self.model.config.get("entity_ranking.tie_handling.rtol")), atol=float(self.model.config.get("entity_ranking.tie_handling.atol"))
#         )
#         is_greater = scores > true_scores.view(-1, 1)
#         num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
#         rank = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)
#         return rank, num_ties

#     def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
#         """Calculates the final rank from (minimum) rank and number of ties.

#         :param rank: batch_size x 1 tensor with number of scores greater than the one of
#         the true score

#         :param num_ties: batch_size x tensor with number of scores equal as the one of
#         the true score

#         :return: batch_size x 1 tensor of ranks

#         """

#         if self.model.config.get("entity_ranking.tie_handling.type") == "rounded_mean_rank":
#             return rank + num_ties // 2
#         elif self.model.config.get("entity_ranking.tie_handling.type") == "best_rank":
#             return rank
#         elif self.model.config.get("entity_ranking.tie_handling.type") == "worst_rank":
#             return rank + num_ties - 1
#         else:
#             raise NotImplementedError

#     def _compute_metrics(self, rank_hist, suffix=""):
#         """Computes desired matrix from rank histogram"""
#         metrics = {}
#         n = torch.sum(rank_hist).item()

#         ranks = torch.arange(1, self.model.dataset.num_entities() + 1).float().to(self.model.config.get("job.device"))
#         metrics["mean_rank" + suffix] = (
#             (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
#         )

#         reciprocal_ranks = 1.0 / ranks
#         metrics["mean_reciprocal_rank" + suffix] = (
#             (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
#         )

#         max_k = min(
#             self.model.dataset.num_entities(),
#             max(self.model.config.get("entity_ranking.hits_at_k_s")),
#         )

#         hits_at_k_s = list(
#                     filter(lambda x: x <= max_k, self.model.config.get("entity_ranking.hits_at_k_s"))
#                 )

#         hits_at_k = (
#             (
#                 torch.cumsum(
#                     rank_hist[: max(hits_at_k_s)], dim=0, dtype=torch.float64
#                 )
#                 / n
#             ).tolist()
#             if n > 0.0
#             else [0.0] * max(hits_at_k_s)
#         )

#         for i, k in enumerate(hits_at_k_s):
#             metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

#         return metrics


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
