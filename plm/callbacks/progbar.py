from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme, MetricsTextColumn, reconfigure, get_console, CustomProgress, pl
from typing import Optional, Any, Iterable, Tuple, List, Dict
from typing_extensions import override
from collections.abc import Generator
import re


class MultiColumnMetricsTextColumn(MetricsTextColumn):
    @override
    def _generate_metrics_texts(self) -> Generator[str, None, None]:
        for name, value in self._metrics.items():
            if not isinstance(value, str):
                value = f"{value:{self._metrics_format}}"
            yield name, value


class MultiColumnTextDelimiter:
    def __init__(self, ncols: int = 5):
        self.ncols = ncols
        self.nrows = None
        self.keys_cache = None

    def _build_node(self, parent: dict, child_string: List[str]) -> Dict[str, dict]:
        node = parent["children"].setdefault(
            child_string[0],
            {
            "prefix": parent["prefix"] + child_string[0],
            "root": parent if parent["root"] is None else parent["root"],
            "parent": parent,
            "children": {},
            "children_squeezed": []
        })
        if len(child_string) == 1:
            node["children_squeezed"].append("")
            node["root"]["leaves"][node["prefix"]] = node
            return parent
        self._build_node(node, child_string[1:])


    def _squeeze(self, node: dict) -> dict:
        if node["root"] is None:
            raise ValueError("Cannot squeeze root node.")
        root = node["root"]
        for name, child in node["children"].items():
            del root["leaves"][child["prefix"]]
            for item in child["children_squeezed"]:
                node["children_squeezed"].append(name + item)
        node["children_squeezed"]
        node["children"] = {}
        root["leaves"][node["prefix"]] = node
        return node


    def _cluster(self, names: List[str], ngroup_minmax: Tuple[int, int]) -> Dict[str, List[str]]:
        prefix_string = [re.findall(r'[a-zA-Z]+|\d+|[^a-zA-Z\d]+', name) for name in names]
        root = {"prefix": "", "root": None, "parent": None, "children": {}, "children_squeezed": [], "leaves": {}}

        # Build tree
        for name in prefix_string:
            self._build_node(root, name)

        # Squeeze leaf nodes
        while len(root["leaves"]) > ngroup_minmax[1] > 1: 
            leaf_groupmax = max(
                [leaf for leaf in root["leaves"].values() if all(len(child["children"]) == 0 for child in leaf["parent"]["children"].values())], # homogeneous leaf node
                key=lambda x: len(x["parent"]["children"]) + len(x["parent"]["children_squeezed"])
                )
            parent_leaf_groupmax = leaf_groupmax["parent"]
            nchildren_max = len(parent_leaf_groupmax["children"])
            if (len(root["leaves"]) - nchildren_max) < ngroup_minmax[0]:
                break
            # Squeeze
            self._squeeze(parent_leaf_groupmax)

        # Squeeze any leaf end with not alphabet
        while len(root["leaves"]) > 1 and len(end_with_notalpha:= [leaf for leaf in root["leaves"].values() if not leaf["prefix"][-1].isalpha()]) > 1:
            self._squeeze(end_with_notalpha[0]["parent"])

        groups = {leaf["prefix"]: [leaf["prefix"] + item for item in leaf["children_squeezed"]] for leaf in root["leaves"].values()}
        return groups
    

    def join(self, items: Iterable[Tuple[str, str]]):
        """Join items with a delimiter for multiple columns."""
        raw_groups = {name: f"{name}: {value}" for name, value in items}
        if len(raw_groups) == 0:
            return ""
        

        sorted_names = sorted(raw_groups.keys())
        if self.keys_cache is None or sorted_names != self.keys_cache or self.nrows is None:

            # Try to reduce it to `ncols` groups.
            self.groups = self._cluster(list(raw_groups.keys()), (max(1, self.ncols - 2), self.ncols))

            # Merge the two groups with the least number of items
            while len(self.groups) > self.ncols: 
                groupname_last1 = min(self.groups, key=lambda k: len(self.groups[k]))
                groupitem_last1 = self.groups.pop(groupname_last1)
                groupname_last2 = min(self.groups, key=lambda k: len(self.groups[k]))
                groupitem_last2 = self.groups.pop(groupname_last2)
                self.groups[f"{groupname_last1} {groupname_last2}"] = groupitem_last1 + groupitem_last2

            self.keys_cache = sorted_names

            # Format table
            self.column_keys = sorted(self.groups.keys(), key = lambda groupname: (len(self.groups[groupname]), groupname),reverse=True)
            self.groups = {key: sorted(self.groups[key], key=lambda x: (len(x), x)) for key in self.column_keys}
            self.ncols = len(self.column_keys)
            self.nrows = max(len(self.groups[key]) for key in self.column_keys)
        
        # Fill table
        table = [["" for _ in range(self.ncols)] for _ in range(self.nrows)]
        col_widths = [0] * self.ncols
        for col_index, key in enumerate(self.column_keys):
            for row_index, name in enumerate(self.groups[key]):
                cell = raw_groups[name]
                table[row_index][col_index] = cell
                
                # Update column width
                col_widths[col_index] = max(col_widths[col_index], len(cell))

        # Build the text representation of the table
        text = ""
        for row in table:
            text += " | ".join(f"{cell:<{col_widths[col]}}" for col, cell in enumerate(row)) + "\n"

        return text


class MultiColumnRichProgressBar(RichProgressBar):
    def __init__(self, metrics_ncols: int = 5):
        refresh_rate: int = 1
        leave: bool = False
        text_delimiter = MultiColumnTextDelimiter(metrics_ncols)
        theme: RichProgressBarTheme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter=text_delimiter,
            metrics_format=".4f",
        )
        console_kwargs: Optional[dict[str, Any]] = None

        super().__init__(refresh_rate, leave, theme, console_kwargs)


    @override
    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MultiColumnMetricsTextColumn( # Use the custom metrics text column
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False