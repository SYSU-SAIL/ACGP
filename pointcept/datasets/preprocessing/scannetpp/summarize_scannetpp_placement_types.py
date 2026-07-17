"""Infer ScanNet++ placement types from instance geometry and support relations.

Every class is allowed on the floor. Additional wall, ceiling, and surface
placement types are enabled when enough training-set instances provide the
corresponding geometric evidence.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.datasets.scannetpp_constants import CLASS_LABELS_PP, INST_LABELS_PP


PLACEMENT_TYPES = ("floor", "wall", "ceiling", "surface")
STRUCTURAL_CLASSES = {"wall", "floor", "ceiling"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/scannetpp")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--output", default="outputs/scannetpp_placement_types/scannetpp_placement_types.json")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--python-output", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--max-contact-points", type=int, default=512)
    parser.add_argument("--tree-voxel-size", type=float, default=0.0)
    parser.add_argument("--wall-distance", type=float, default=0.08)
    parser.add_argument("--ceiling-distance", type=float, default=0.10)
    parser.add_argument("--floor-distance", type=float, default=0.10)
    parser.add_argument("--min-contact-ratio", type=float, default=0.03)
    parser.add_argument("--wall-min-contact-ratio", type=float, default=0.20)
    parser.add_argument("--ceiling-min-contact-ratio", type=float, default=0.10)
    parser.add_argument("--surface-min-gap", type=float, default=-0.05)
    parser.add_argument("--surface-max-gap", type=float, default=0.15)
    parser.add_argument("--surface-min-overlap", type=float, default=0.15)
    parser.add_argument("--min-evidence-count", type=int, default=2)
    parser.add_argument("--min-evidence-ratio", type=float, default=0.05)
    return parser.parse_args()


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def contact_ratio(points, tree, distance, max_points):
    if tree is None or len(points) == 0:
        return 0.0
    points = sample_points(points, max_points)
    nearest, _ = tree.query(points, k=1)
    return float(np.mean(nearest <= distance))


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points
    voxel = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    return points[indices]


def make_tree(points, voxel_size=0.0):
    points = voxel_downsample(points, voxel_size)
    return cKDTree(points) if len(points) > 0 else None


def instance_class_id(segment, mask):
    labels = segment[mask]
    labels = labels[(labels >= 0) & (labels < len(CLASS_LABELS_PP))]
    if len(labels) == 0:
        return None
    counts = np.bincount(labels, minlength=len(CLASS_LABELS_PP))
    return int(np.argmax(counts))


def build_instances(coord, segment, instance):
    instances = []
    for instance_id in np.unique(instance):
        if instance_id < 0:
            continue
        mask = instance == instance_id
        class_id = instance_class_id(segment, mask)
        if class_id is None:
            continue
        points = coord[mask]
        if len(points) == 0:
            continue
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        instances.append(
            {
                "id": int(instance_id),
                "class_id": class_id,
                "class_name": CLASS_LABELS_PP[class_id],
                "points": points,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        )
    return instances


def xy_overlap_ratio(target, support):
    target_min, target_max = target["bbox_min"], target["bbox_max"]
    support_min, support_max = support["bbox_min"], support["bbox_max"]
    overlap = np.maximum(
        np.minimum(target_max[:2], support_max[:2])
        - np.maximum(target_min[:2], support_min[:2]),
        0.0,
    )
    target_size = np.maximum(target_max[:2] - target_min[:2], 1e-4)
    return float(np.prod(overlap) / np.prod(target_size))


def find_surface_support(target, instances, min_gap, max_gap, min_overlap):
    candidates = []
    target_bottom = float(target["bbox_min"][2])
    for support in instances:
        if support["id"] == target["id"]:
            continue
        if support["class_name"] in STRUCTURAL_CLASSES:
            continue
        support_top = float(support["bbox_max"][2])
        gap = target_bottom - support_top
        if gap < min_gap or gap > max_gap:
            continue
        overlap = xy_overlap_ratio(target, support)
        if overlap < min_overlap:
            continue
        candidates.append((abs(gap), -overlap, support))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def make_stats():
    return defaultdict(
        lambda: {
            "instances": 0,
            "evidence": Counter({placement_type: 0 for placement_type in PLACEMENT_TYPES}),
            "surface_support_classes": Counter(),
        }
    )


def update_scene_stats(scene_dir, stats, args):
    coord = np.load(scene_dir / "coord.npy").astype(np.float32)
    segment = np.load(scene_dir / "segment.npy")
    instance = np.load(scene_dir / "instance.npy")
    if segment.ndim > 1:
        segment = segment[:, 0]
    if instance.ndim > 1:
        instance = instance[:, 0]
    segment = segment.reshape(-1).astype(np.int32)
    instance = instance.reshape(-1).astype(np.int32)
    if not (len(coord) == len(segment) == len(instance)):
        raise ValueError(f"Mismatched arrays in {scene_dir}")
    if not np.all(np.isfinite(coord)):
        raise ValueError(f"Non-finite coordinates in {scene_dir}")

    wall_id = CLASS_LABELS_PP.index("wall")
    floor_id = CLASS_LABELS_PP.index("floor")
    ceiling_id = CLASS_LABELS_PP.index("ceiling")
    wall_tree = make_tree(coord[segment == wall_id], args.tree_voxel_size)
    floor_tree = make_tree(coord[segment == floor_id], args.tree_voxel_size)
    ceiling_tree = make_tree(coord[segment == ceiling_id], args.tree_voxel_size)
    instances = build_instances(coord, segment, instance)

    for target in instances:
        name = target["class_name"]
        points = target["points"]
        bbox_min, bbox_max = target["bbox_min"], target["bbox_max"]
        height = max(float(bbox_max[2] - bbox_min[2]), 1e-4)
        bottom_points = points[points[:, 2] <= bbox_min[2] + max(0.05, 0.15 * height)]
        top_points = points[points[:, 2] >= bbox_max[2] - max(0.05, 0.15 * height)]

        stats[name]["instances"] += 1
        if name in STRUCTURAL_CLASSES:
            continue
        floor_contact = (
            contact_ratio(bottom_points, floor_tree, args.floor_distance, args.max_contact_points)
            >= args.min_contact_ratio
        )
        if floor_contact:
            stats[name]["evidence"]["floor"] += 1
        if (
            not floor_contact
            and name != "wall"
            and contact_ratio(points, wall_tree, args.wall_distance, args.max_contact_points)
            >= args.wall_min_contact_ratio
        ):
            stats[name]["evidence"]["wall"] += 1
        if (
            not floor_contact
            and name != "ceiling"
            and contact_ratio(top_points, ceiling_tree, args.ceiling_distance, args.max_contact_points)
            >= args.ceiling_min_contact_ratio
        ):
            stats[name]["evidence"]["ceiling"] += 1

        support = find_surface_support(
            target,
            instances,
            args.surface_min_gap,
            args.surface_max_gap,
            args.surface_min_overlap,
        )
        if support is not None:
            stats[name]["evidence"]["surface"] += 1
            stats[name]["surface_support_classes"][support["class_name"]] += 1


def summarize_scene(scene_dir, args):
    stats = make_stats()
    update_scene_stats(scene_dir, stats, args)
    return {
        name: {
            "instances": values["instances"],
            "evidence": values["evidence"],
            "surface_support_classes": values["surface_support_classes"],
        }
        for name, values in stats.items()
    }


def merge_stats(stats, scene_stats):
    for name, values in scene_stats.items():
        stats[name]["instances"] += values["instances"]
        stats[name]["evidence"].update(values["evidence"])
        stats[name]["surface_support_classes"].update(
            values["surface_support_classes"]
        )


def infer_types(stats, args):
    result = {}
    for name in INST_LABELS_PP:
        count = stats[name]["instances"]
        supported = ["floor"]
        if name in STRUCTURAL_CLASSES:
            result[name] = supported
            continue
        for placement_type in ("wall", "ceiling", "surface"):
            evidence = stats[name]["evidence"][placement_type]
            ratio = evidence / count if count else 0.0
            if evidence >= args.min_evidence_count and ratio >= args.min_evidence_ratio:
                supported.append(placement_type)
        result[name] = supported
    return result


def serialize_stats(stats):
    output = {}
    for name in INST_LABELS_PP:
        count = stats[name]["instances"]
        evidence = dict(stats[name]["evidence"])
        output[name] = {
            "instances": count,
            "evidence": evidence,
            "evidence_ratio": {
                placement_type: (evidence[placement_type] / count if count else 0.0)
                for placement_type in PLACEMENT_TYPES
            },
            "surface_support_classes": dict(
                stats[name]["surface_support_classes"].most_common()
            ),
        }
    return output


def write_csv(path, placement_types, class_stats):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_name",
                "supported_types",
                "instances",
                "floor_evidence",
                "wall_evidence",
                "ceiling_evidence",
                "surface_evidence",
                "surface_support_classes",
            ]
        )
        for name in INST_LABELS_PP:
            stats = class_stats[name]
            writer.writerow(
                [
                    name,
                    "|".join(placement_types[name]),
                    stats["instances"],
                    stats["evidence"]["floor"],
                    stats["evidence"]["wall"],
                    stats["evidence"]["ceiling"],
                    stats["evidence"]["surface"],
                    json.dumps(stats["surface_support_classes"], sort_keys=True),
                ]
            )


def write_python(path, placement_types):
    with path.open("w") as file:
        file.write('"""Generated by summarize_scannetpp_placement_types.py."""\n\n')
        file.write("SCANNETPP_PLACEMENT_TYPES = {\n")
        for name in INST_LABELS_PP:
            file.write(f"    {name!r}: {tuple(placement_types[name])!r},\n")
        file.write("}\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    scene_dirs = []
    for split in args.splits:
        scene_dirs.extend(
            path for path in sorted((data_root / split).iterdir())
            if path.is_dir()
        )
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: args.max_scenes]
    if not scene_dirs:
        raise ValueError(f"No scenes found under {data_root} for splits={args.splits}")

    stats = make_stats()
    if args.num_workers <= 1:
        scene_stats_iter = map(summarize_scene, scene_dirs, repeat(args))
        for index, scene_stats in enumerate(scene_stats_iter, start=1):
            merge_stats(stats, scene_stats)
            if index % 50 == 0 or index == len(scene_dirs):
                print(f"Processed {index}/{len(scene_dirs)} scenes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            scene_stats_iter = executor.map(
                summarize_scene,
                scene_dirs,
                repeat(args),
                chunksize=args.chunksize,
            )
            for index, scene_stats in enumerate(scene_stats_iter, start=1):
                merge_stats(stats, scene_stats)
                if index % 50 == 0 or index == len(scene_dirs):
                    print(f"Processed {index}/{len(scene_dirs)} scenes", flush=True)

    placement_types = infer_types(stats, args)
    class_stats = serialize_stats(stats)
    output = Path(args.output)
    csv_output = Path(args.csv_output) if args.csv_output else output.with_suffix(".csv")
    python_output = Path(args.python_output) if args.python_output else output.with_suffix(".py")
    for path in (output, csv_output, python_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "data_root": str(data_root),
            "splits": args.splits,
            "scenes": len(scene_dirs),
            "num_workers": args.num_workers,
            "placement_types": list(PLACEMENT_TYPES),
            "all_classes_support_floor": True,
            "thresholds": {
                "wall_distance": args.wall_distance,
                "tree_voxel_size": args.tree_voxel_size,
                "ceiling_distance": args.ceiling_distance,
                "floor_distance": args.floor_distance,
                "min_contact_ratio": args.min_contact_ratio,
                "wall_min_contact_ratio": args.wall_min_contact_ratio,
                "ceiling_min_contact_ratio": args.ceiling_min_contact_ratio,
                "surface_min_gap": args.surface_min_gap,
                "surface_max_gap": args.surface_max_gap,
                "surface_min_overlap": args.surface_min_overlap,
                "min_evidence_count": args.min_evidence_count,
                "min_evidence_ratio": args.min_evidence_ratio,
            },
        },
        "placement_types": placement_types,
        "class_stats": class_stats,
    }
    with output.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    write_csv(csv_output, placement_types, class_stats)
    write_python(python_output, placement_types)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")
    print(f"Wrote {python_output}")


if __name__ == "__main__":
    main()
