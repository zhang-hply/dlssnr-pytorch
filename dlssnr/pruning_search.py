"""Deterministic depth/width searches sharing physical inference measurements.

Each method owns its logical evaluation budget. A cache hit still consumes a
method's first request for that structure, but never repeats network inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .ablation import layer_macs, units
from .pruning import PruningSpec, WidthSpec, layouts, validate_spec


def spec_key(spec: PruningSpec) -> tuple:
    return (spec.dropped_blocks, tuple((width.block, width.mlp or (), width.heads or ())
                                      for width in spec.widths))


def _changes(spec: PruningSpec) -> int:
    return len(spec.dropped_blocks) + sum((width.mlp is not None) + (width.heads is not None)
                                          for width in spec.widths)


def best_rows(rows, top_k: int = 10) -> list[dict]:
    def key(row):
        spec = PruningSpec.from_dict(row["pruning_spec"])
        return row["mse"], -row["macs"], _changes(spec), spec_key(spec)
    return sorted((row for row in rows if row["feasible"]), key=key)[:top_k]


@dataclass(frozen=True)
class SearchSite:
    block: int
    kind: str
    # Reduced counts in decreasing order; full size is implicit.
    levels: tuple[int, ...] = ()


def search_sites(plan, method: str) -> tuple[SearchSite, ...]:
    if method not in {"depth", "width", "hybrid"}:
        raise ValueError(f"unknown pruning method: {method}")
    result = []
    for block, layout in sorted(layouts(plan).items()):
        if method != "width":
            result.append(SearchSite(block, "whole"))
        if method != "depth":
            hidden = layout.hidden_size
            levels = tuple(sorted({min(hidden, math.ceil(hidden * ratio / 32) * 32)
                                   for ratio in (.75, .5)} - {hidden}, reverse=True))
            if levels:
                result.append(SearchSite(block, "mlp", levels))
            heads = layout.heads
            levels = tuple(sorted({max(1, math.ceil(heads * ratio)) for ratio in (.75, .5)}
                                  - {heads}, reverse=True))
            if levels:
                result.append(SearchSite(block, "head", levels))
    return tuple(result)


class EvaluationCache:
    """One measurement per canonical architecture across all methods."""

    def __init__(self, plan, keep_ratio):
        self.plan, self.keep_ratio = plan, keep_ratio
        self.results: dict[PruningSpec, dict] = {}
        self.evaluate = None

    def __call__(self, spec):
        if spec in self.results:
            return self.results[spec]
        stats = validate_spec(self.plan, spec, keep_ratio=1.0)
        if self.evaluate is None:
            raise RuntimeError("no pruning evaluator has been configured")
        started = time.perf_counter()
        metrics = self.evaluate(spec)
        mse = float(metrics["mse"])
        if not math.isfinite(mse) or mse < 0:
            raise ValueError(f"invalid pruning MSE: {mse}")
        row = dict(metrics)
        row.update(pruning_spec=spec.to_dict(), macs=stats["macs"],
                   saved_macs=stats["total_macs"] - stats["macs"],
                   keep_ratio=stats["keep_ratio"],
                   feasible=stats["macs"] <= stats["total_macs"] * self.keep_ratio,
                   mse=mse, psnr_db=math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse),
                   evaluation_seconds=time.perf_counter() - started)
        self.results[spec] = row
        return row


class WidthSearch:
    """Width or hybrid beam search over nested, teacher-ranked dimensions."""

    def __init__(self, plan, config, method: str):
        if method not in {"width", "hybrid"}:
            raise ValueError("WidthSearch requires width or hybrid")
        if not math.isfinite(config.keep_ratio) or not 0 < config.keep_ratio < 1:
            raise ValueError("--keep-ratio must be between 0 and 1 (exclusive)")
        if config.beam_width < 1 or config.expand_top_k < 1:
            raise ValueError("--beam-width and --expand-top-k must be positive")
        self.plan, self.config, self.method = plan, config, method
        self.layouts = layouts(plan)
        self.sites = search_sites(plan, method)
        if not self.sites:
            raise ValueError(f"{method}: no eligible pruning sites")
        if config.max_evals < len(self.sites) + 1:
            raise ValueError(f"{method}: --max-evals must be at least {len(self.sites) + 1} "
                             "for first-level screening and a feasible seed")
        self.total_macs = sum(layer_macs(plan).values())
        self.results: dict[PruningSpec, dict] = {}
        self.stop_reason = self.phase = "not_started"
        self.rankings = {block: {"mlp": tuple(tuple(range(layout.hidden_size))
                                             for _ in range(layout.groups)),
                                 "heads": tuple(range(layout.heads))}
                         for block, layout in self.layouts.items()}
        terminal = PruningSpec()
        for site in self.sites:
            while (child := self.next_spec(terminal, site)) is not None:
                terminal = child
        if not self.feasible(terminal):
            remaining = self.cost(terminal) / self.total_macs
            raise ValueError(f"{method}: minimum remaining MAC ratio is {remaining:.6f}; "
                             f"cannot reach --keep-ratio {config.keep_ratio}")

    def set_rankings(self, rankings) -> None:
        if self.results:
            raise ValueError("cannot change width rankings after search starts")
        checked = {}
        for block, layout in self.layouts.items():
            try:
                mlp = tuple(tuple(group) for group in rankings[block]["mlp"])
                heads = tuple(rankings[block]["heads"])
            except (KeyError, TypeError) as error:
                raise ValueError(f"missing teacher width ranking for block {block}") from error
            if (len(mlp) != layout.groups
                    or any(any(type(index) is not int for index in group)
                           or sorted(group) != list(range(layout.hidden_size)) for group in mlp)
                    or any(type(index) is not int for index in heads)
                    or sorted(heads) != list(range(layout.heads))):
                raise ValueError(f"invalid teacher width ranking for block {block}")
            checked[block] = {"mlp": mlp, "heads": heads}
        self.rankings = checked

    def cost(self, spec) -> int:
        return validate_spec(self.plan, spec, keep_ratio=1.0)["macs"]

    def feasible(self, spec) -> bool:
        return self.cost(spec) <= self.total_macs * self.config.keep_ratio

    def next_spec(self, spec: PruningSpec, site: SearchSite) -> PruningSpec | None:
        if site.block in spec.dropped_blocks:
            return None
        widths = {width.block: width for width in spec.widths}
        if site.kind == "whole":
            widths.pop(site.block, None)
            return PruningSpec(tuple(sorted((*spec.dropped_blocks, site.block))),
                               tuple(widths[block] for block in sorted(widths)))
        previous = widths.get(site.block)
        mlp = previous.mlp if previous is not None else None
        heads = previous.heads if previous is not None else None
        layout = self.layouts[site.block]
        current = ((len(mlp[0]) if mlp is not None else layout.hidden_size) if site.kind == "mlp"
                   else (len(heads) if heads is not None else layout.heads))
        count = next((level for level in site.levels if level < current), None)
        if count is None:
            return None
        if site.kind == "mlp":
            mlp = tuple(tuple(sorted(group[:count])) for group in self.rankings[site.block]["mlp"])
        else:
            heads = tuple(sorted(self.rankings[site.block]["heads"][:count]))
        widths[site.block] = WidthSpec(site.block, mlp, heads)
        return PruningSpec(spec.dropped_blocks, tuple(widths[block] for block in sorted(widths)))

    def best_results(self, top_k=10):
        return best_rows(self.results.values(), top_k)

    def run(self, fetch, on_result=None, on_progress=None):
        def progress(message):
            if on_progress is not None:
                on_progress(message)

        def measure(spec):
            if spec in self.results:
                return self.results[spec]
            if len(self.results) >= self.config.max_evals:
                raise RuntimeError("internal method evaluation budget exceeded")
            progress(f"[{len(self.results) + 1}/{self.config.max_evals}] {self.method} "
                     f"{self.phase}: starting keep={self.cost(spec) / self.total_macs:.2%}")
            row = fetch(spec)
            self.results[spec] = row
            if on_result is not None:
                on_result(self, row)
            return row

        self.stop_reason, self.phase = "running", "single_actions"
        empty = PruningSpec()
        progress(f"{self.method}: first-level sensitivity screening for {len(self.sites)} sites")
        for site in self.sites:
            measure(self.next_spec(empty, site))
        kinds = {"whole": 0, "mlp": 1, "head": 2}

        def priority(site):
            row = self.results[self.next_spec(empty, site)]
            return row["mse"] / row["saved_macs"], row["mse"], site.block, kinds[site.kind]

        ranking = sorted(self.sites, key=priority)
        self.phase = "seed"
        seed = empty
        for site in ranking:
            while (child := self.next_spec(seed, site)) is not None:
                seed = child
                if self.feasible(seed):
                    break
            if self.feasible(seed):
                break
        if not self.feasible(seed):
            raise RuntimeError("internal pruning seed failed to reach a reachable budget")
        measure(seed)

        self.phase = "beam"
        beam, expanded, depth = [empty], set(), 0
        while beam:
            depth += 1
            children = set()
            for parent in beam:
                expanded.add(parent)
                additions = []
                for site in ranking:
                    child = self.next_spec(parent, site)
                    if child is not None and child not in expanded:
                        additions.append(child)
                        if len(additions) == self.config.expand_top_k:
                            break
                children.update(additions)
            children.difference_update(expanded)
            progress(f"{self.method} beam depth={depth}: paths={len(beam)}, "
                     f"combinations={len(children)}, "
                     f"new={sum(child not in self.results for child in children)}, "
                     f"evaluated={len(self.results)}/{self.config.max_evals}")
            next_rows = []
            for child in sorted(children, key=spec_key):
                if child not in self.results and len(self.results) >= self.config.max_evals:
                    self.stop_reason, self.phase = "max_evals", "done"
                    return
                row = measure(child)
                if not row["feasible"]:
                    next_rows.append((child, row))
            next_rows.sort(key=lambda item: (item[1]["mse"] / item[1]["saved_macs"],
                                             item[1]["mse"], -item[1]["macs"], spec_key(item[0])))
            beam = [spec for spec, _ in next_rows[:self.config.beam_width]]
        self.stop_reason, self.phase = "search_exhausted", "done"


class DepthSearch:
    """Adapt the original depth implementation without changing its traversal."""

    method = "depth"

    def __init__(self, plan, config, factory):
        self.inner = factory(units(plan), sum(layer_macs(plan).values()), config)
        self.config = config
        self.results: dict[PruningSpec, dict] = {}

    @property
    def phase(self):
        return self.inner.phase

    @property
    def stop_reason(self):
        return self.inner.stop_reason

    def best_results(self, top_k=10):
        return best_rows(self.results.values(), top_k)

    def run(self, fetch, on_result=None, on_progress=None):
        pending = {}

        def evaluate(blocks):
            spec = PruningSpec(tuple(blocks))
            pending[spec] = fetch(spec)
            return pending[spec]

        def record(_state, row):
            spec = PruningSpec(tuple(row["blocks"]))
            self.results[spec] = pending[spec]
            if on_result is not None:
                on_result(self, pending[spec])

        self.inner.run(evaluate, record, on_progress)


class SearchSuite:
    """Run selected methods, each with its own budget and shared evaluations."""

    def __init__(self, plan, config, method, depth_factory):
        names = ("depth", "width", "hybrid") if method == "compare" else (method,)
        self.methods = {name: (DepthSearch(plan, config, depth_factory) if name == "depth"
                               else WidthSearch(plan, config, name)) for name in names}
        self.cache = EvaluationCache(plan, config.keep_ratio)
        self.total_macs = sum(layer_macs(plan).values())
        self.phase = self.stop_reason = "not_started"
        self.active_method = None

    def set_rankings(self, rankings):
        for search in self.methods.values():
            if isinstance(search, WidthSearch):
                search.set_rankings(rankings)

    def run(self, evaluate, on_result=None, on_progress=None):
        self.cache.evaluate = evaluate
        self.phase, self.stop_reason = "search", "running"
        for name, search in self.methods.items():
            self.active_method = name
            if on_progress is not None:
                on_progress(f"Starting {name}: independent budget={search.config.max_evals}; "
                            f"shared measurements={len(self.cache.results)}")
            search.run(self.cache, on_result, on_progress)
        self.active_method = None
        self.stop_reason = ("max_evals" if any(search.stop_reason == "max_evals"
                                              for search in self.methods.values()) else "search_exhausted")
        self.phase = "done"

    def best_results(self, top_k=10):
        return best_rows(self.cache.results.values(), top_k)

    def snapshot(self, top_k=10):
        methods = {}
        for name, search in self.methods.items():
            top = search.best_results(top_k)
            methods[name] = {"best": top[0] if top else None, "top_results": top,
                             "stop_reason": search.stop_reason,
                             "progress": {"unique_evaluations": len(search.results),
                                          "max_evals": search.config.max_evals,
                                          "phase": search.phase}}
        return methods

    def ssim_results(self, top_k=10):
        chosen = {}
        for search in self.methods.values():
            for row in search.best_results(top_k):
                chosen[PruningSpec.from_dict(row["pruning_spec"])] = row
        return [chosen[spec] for spec in sorted(chosen, key=spec_key)]
