import multiprocessing as mp


class SharedRuntimeStats:
    """Small multiprocessing-safe counter for DataLoader worker timings."""

    def __init__(self, name):
        self.name = name
        self._lock = mp.Lock()
        self._total_time = mp.Value("d", 0.0, lock=False)
        self._count = mp.Value("q", 0, lock=False)

    def add(self, elapsed):
        with self._lock:
            self._total_time.value += float(elapsed)
            self._count.value += 1

    def snapshot(self):
        with self._lock:
            return {
                "name": self.name,
                "count": int(self._count.value),
                "total_time": float(self._total_time.value),
            }

    def reset(self):
        with self._lock:
            self._total_time.value = 0.0
            self._count.value = 0


def collect_runtime_stats(dataset):
    """Recursively collect SharedRuntimeStats from datasets and transforms."""
    stats = []
    seen = set()

    def add_stat(stat):
        if stat is None:
            return
        stat_id = id(stat)
        if stat_id not in seen:
            stats.append(stat)
            seen.add(stat_id)

    def visit(obj):
        if obj is None:
            return
        add_stat(getattr(obj, "runtime_stats", None))

        transform = getattr(obj, "transform", None)
        if transform is not None:
            visit(transform)

        transforms = getattr(obj, "transforms", None)
        if transforms is not None:
            for t in transforms:
                visit(t)

        datasets = getattr(obj, "datasets", None)
        if datasets is not None:
            for ds in datasets:
                visit(ds)

        wrapped = getattr(obj, "dataset", None)
        if wrapped is not None and wrapped is not obj:
            visit(wrapped)

    visit(dataset)
    return stats
