from typing import List, Callable, Tuple
from os import PathLike

def get_rule_counts(fnamelist: list, rule: Callable) -> Tuple[dict, dict]:
    groups = {}
    for fname in fnamelist:
        groupname = rule(fname)
        groups.setdefault(groupname, []).append(fname)
    return groups, {k: len(v) for k, v in groups.items()}

def base_rule(fname: str) -> str:
    basename = fname.split("/")[-1]
    groupname = "@".join(basename.split("@")[:-2])
    return groupname

def acq_rule(fname: str) -> str:
    basename = fname.split("/")[-1]
    groupname = "@".join(basename.split("@")[0:1])
    return groupname

def center_rule(fname: str) -> str:
    basename = fname.split("/")[-1]
    groupname = "@".join(basename.split("@")[3:4])
    return groupname

def balance_weights(weights: dict, rules: List[Tuple[Callable, dict]], max_weight: int = 4, min_weight:int = 1, momentum: float = 1.) -> dict:
    nsample = len(weights) # Global number of samples
    for fname, weight in weights.items():
        for rule, counts in rules:
            ngroup = len(counts)
            meanvalue = nsample / ngroup
            samplevalue = counts[rule(fname)] # Number of samples in the group to which the sample belongs

            ratio = meanvalue / samplevalue
            ratio = (ratio - 1) * momentum + 1

            weight = weight * ratio
        
        weight = max(min_weight, min(max_weight, weight)) # Clip the weight to the range [min_weight, max_weight]
        weight = round(weight)
        weights[fname] = weight
    return weights

def weight_sum_for_groups(weights: dict, rule: Callable) -> dict:
    groups = {}
    group_weights = {}
    for fname, weight in weights.items():
        groupname = rule(fname)
        groups.setdefault(groupname, []).append(fname)
        group_weights.setdefault(groupname, 0)
        group_weights[groupname] += weight
    return groups, group_weights

class AtSplitedBalanceSampler:
    def __init__(self, max_weight: int = 4, min_weight: int = 1, momentum: float = 1.0):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.momentum = momentum

    def __call__(self, fname: List[PathLike | str]):
        weights = {f: 1.0 for f in fname}
        rules = [
            (base_rule, get_rule_counts(fname, base_rule)[1]),
            (acq_rule, get_rule_counts(fname, acq_rule)[1]),
            (center_rule, get_rule_counts(fname, center_rule)[1])
        ]
        weights = balance_weights(weights, rules, self.max_weight, self.min_weight, self.momentum)
        
        output_fnamelist = [f for f, w in weights.items() for _ in range(w) ]
        return output_fnamelist