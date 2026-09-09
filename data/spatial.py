import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
import yaml
import numpy as np

from utils.general import colorstr


def parse_sample_spatial_key(stem: str) -> Tuple[str, Optional[int]]:
    """
    Extract geographic region identifier and sequential tile index from sample stem.

    Supported patterns:
    - 'Chainage_17_00156' -> ('Chainage_17', 156)
    - 'Section_2_tile_045' -> ('Section_2', 45)
    - 'SiteA_0034' -> ('SiteA', 34)
    - '12.34_56.78' -> ('12.34_56.78', None)

    Args:
        stem (str): Sample filename stem (without extension)

    Returns:
        region (str): Region or corridor section identifier
        tile_idx (int or None): Sequential tile integer index if available
    """
    # Pattern 1: Chainage / Section linear corridor: Prefix_RegionNumber_Index
    m_chainage = re.match(r'^(Chainage_\d+|Section_\d+|Site_[A-Za-z0-9]+)_(\d+)$', stem, re.IGNORECASE)
    if m_chainage:
        return m_chainage.group(1), int(m_chainage.group(2))

    # Pattern 2: Name with trailing numeric index: Prefix_12345
    m_general = re.match(r'^(.*?)[_-](\d+)$', stem)
    if m_general:
        return m_general.group(1), int(m_general.group(2))

    # Fallback: Whole stem as individual region
    return stem, None


def audit_spatial_splits(dataset_dir_or_yaml: Union[str, Path, Dict],
                         train_split: str = 'train',
                         val_split: str = 'validation',
                         test_split: Optional[str] = 'test',
                         verbose: bool = True) -> Dict:
    """
    Audit dataset train/val/test splits for geographic proximity and spatial data leakage.

    Checks:
    1. Region-level leakage: Are samples from the same geographic region / corridor present in both train and validation?
    2. Tile proximity / overlap: Are adjacent sequential linear patches split across train and validation?
    3. Spatial balance: Distribution of samples across regions per split.

    Args:
        dataset_dir_or_yaml: Path to dataset YAML or root dataset directory
        train_split: Name of training folder
        val_split: Name of validation folder
        test_split: Name of test folder (optional)
        verbose: Whether to print formatted audit report

    Returns:
        audit_results (dict): Structured audit findings and metrics
    """
    # Resolve root path
    if isinstance(dataset_dir_or_yaml, (str, Path)):
        p = Path(dataset_dir_or_yaml)
        if p.suffix in ['.yaml', '.yml'] and p.exists():
            with open(p, 'r') as f:
                cfg = yaml.safe_load(f)
            root_dir = Path(cfg.get('path', p.parent))
            train_split = cfg.get('train', train_split)
            val_split = cfg.get('val', val_split)
            test_split = cfg.get('test', test_split)
        else:
            root_dir = p
    elif isinstance(dataset_dir_or_yaml, dict):
        root_dir = Path(dataset_dir_or_yaml.get('path', '.'))
        train_split = dataset_dir_or_yaml.get('train', train_split)
        val_split = dataset_dir_or_yaml.get('val', val_split)
        test_split = dataset_dir_or_yaml.get('test', test_split)
    else:
        raise TypeError(f"Invalid dataset_dir_or_yaml type: {type(dataset_dir_or_yaml)}")

    splits = {'train': train_split, 'val': val_split}
    if test_split and (root_dir / test_split).exists():
        splits['test'] = test_split

    samples_by_split: Dict[str, List[str]] = {}
    regions_by_split: Dict[str, Dict[str, List[int]]] = {}

    for split_name, split_folder in splits.items():
        split_path = root_dir / split_folder
        samples_by_split[split_name] = []
        regions_by_split[split_name] = {}

        if not split_path.exists():
            continue

        # Look for images or masks in subfolders or directly
        cand_dirs = [split_path / 'IMAGE', split_path / 'LABEL', split_path]
        sample_files = []
        for cd in cand_dirs:
            if cd.exists() and cd.is_dir():
                found = glob.glob(str(cd / '*.*'))
                if found:
                    sample_files = found
                    break

        stems = sorted(list(set(Path(f).stem for f in sample_files)))
        samples_by_split[split_name] = stems

        for s in stems:
            region, idx = parse_sample_spatial_key(s)
            if region not in regions_by_split[split_name]:
                regions_by_split[split_name][region] = []
            if idx is not None:
                regions_by_split[split_name][region].append(idx)

    # Compute overlap and spatial leakage metrics
    train_regions = set(regions_by_split.get('train', {}).keys())
    val_regions = set(regions_by_split.get('val', {}).keys())
    test_regions = set(regions_by_split.get('test', {}).keys())

    leakage_train_val = train_regions.intersection(val_regions)
    leakage_train_test = train_regions.intersection(test_regions)
    leakage_val_test = val_regions.intersection(test_regions)

    has_spatial_leakage = len(leakage_train_val) > 0 or len(leakage_train_test) > 0

    # Detailed proximity check for shared regions
    proximity_warnings = []
    if leakage_train_val:
        for shared_reg in leakage_train_val:
            tr_indices = set(regions_by_split['train'][shared_reg])
            val_indices = set(regions_by_split['val'][shared_reg])
            if tr_indices and val_indices:
                min_dist = min(abs(t - v) for t in tr_indices for v in val_indices)
                proximity_warnings.append({
                    'region': shared_reg,
                    'train_count': len(tr_indices),
                    'val_count': len(val_indices),
                    'min_tile_distance': min_dist
                })

    results = {
        'root_dir': str(root_dir),
        'total_samples': {k: len(v) for k, v in samples_by_split.items()},
        'regions_by_split': {k: list(v.keys()) for k, v in regions_by_split.items()},
        'leakage_train_val': list(leakage_train_val),
        'leakage_train_test': list(leakage_train_test),
        'leakage_val_test': list(leakage_val_test),
        'proximity_warnings': proximity_warnings,
        'has_spatial_leakage': has_spatial_leakage,
        'status': 'FAIL: SPATIAL LEAKAGE DETECTED' if has_spatial_leakage else 'PASS: REGIONALLY ISOLATED SPLITS'
    }

    if verbose:
        print("\n" + "=" * 80)
        print(colorstr('bold', 'cyan', "[GEOSPATIAL AUDIT] SPATIAL DATA LEAKAGE & SPLIT AUDIT REPORT"))
        print("=" * 80)
        print(f"  Dataset Root Directory : {root_dir}")
        for s_name, count in results['total_samples'].items():
            regs = results['regions_by_split'].get(s_name, [])
            print(f"  Split '{s_name:10s}'     : {count:4d} samples across {len(regs)} regions {regs}")
        print("-" * 80)

        if not has_spatial_leakage:
            print(colorstr('bright_green', "  [STATUS: PASS] Zero regional overlap detected between splits."))
            print("  Training, validation, and test sets are partitioned by independent geographic regions.")
            print("  Spatial autocorrelation leakage between train and validation is safely prevented.")
        else:
            print(colorstr('bold', 'bright_red', "  [STATUS: FAIL] SPATIAL DATA LEAKAGE DETECTED!"))
            if leakage_train_val:
                print(colorstr('bright_red', f"  Direct region overlap (Train <-> Val): {list(leakage_train_val)}"))
            if leakage_train_test:
                print(colorstr('bright_red', f"  Direct region overlap (Train <-> Test): {list(leakage_train_test)}"))
            if proximity_warnings:
                print("  Geographic Proximity Violations:")
                for pw in proximity_warnings:
                    print(f"    - Region '{pw['region']}': Train ({pw['train_count']} tiles) and Val ({pw['val_count']} tiles) share linear corridor with min step distance = {pw['min_tile_distance']}.")
            print("\n  Recommendation: Re-partition dataset using region-based or spatial-block splitting.")
            print("  Keep entire corridor sections / catchment basins in either train OR validation, never both.")
        print("=" * 80 + "\n")

    return results


def create_spatial_region_splits(samples_or_dir: Union[str, Path, List[str]],
                                 val_regions: Optional[List[str]] = None,
                                 test_regions: Optional[List[str]] = None,
                                 val_ratio: float = 0.2,
                                 seed: int = 42) -> Dict[str, List[str]]:
    """
    Partition samples geographically by whole region / corridor section into train/val/test splits.

    Guarantees zero geographic overlap / spatial leakage between splits.

    Args:
        samples_or_dir: Directory containing sample files or list of sample stems
        val_regions: Explicit list of region names to designate for validation (e.g. ['Chainage_17'])
        test_regions: Explicit list of region names to designate for test (optional)
        val_ratio: Target proportion of samples in validation split if val_regions is None
        seed: Random seed for deterministic region assignment

    Returns:
        splits (dict): {'train': [stems], 'val': [stems], 'test': [stems]}
    """
    if isinstance(samples_or_dir, (str, Path)):
        p = Path(samples_or_dir)
        files = glob.glob(str(p / '**/*.*'), recursive=True)
        stems = sorted(list(set(Path(f).stem for f in files if Path(f).suffix.lower() in ['.png', '.tif', '.jpg'])))
    else:
        stems = sorted(list(set(samples_or_dir)))

    # Group stems by region
    region_map: Dict[str, List[str]] = {}
    for s in stems:
        region, _ = parse_sample_spatial_key(s)
        if region not in region_map:
            region_map[region] = []
        region_map[region].append(s)

    all_regions = sorted(list(region_map.keys()))
    val_regions_set = set(val_regions) if val_regions else set()
    test_regions_set = set(test_regions) if test_regions else set()

    # If explicit regions not provided, select regions greedily to match target val_ratio
    if not val_regions_set:
        rng = np.random.RandomState(seed)
        shuffled_regions = list(all_regions)
        rng.shuffle(shuffled_regions)

        target_val_count = int(len(stems) * val_ratio)
        current_val_count = 0

        for r in shuffled_regions:
            if current_val_count < target_val_count and r not in test_regions_set:
                val_regions_set.add(r)
                current_val_count += len(region_map[r])

    train_stems = []
    val_stems = []
    test_stems = []

    for r, r_stems in region_map.items():
        if r in test_regions_set:
            test_stems.extend(r_stems)
        elif r in val_regions_set:
            val_stems.extend(r_stems)
        else:
            train_stems.extend(r_stems)

    return {
        'train': sorted(train_stems),
        'val': sorted(val_stems),
        'test': sorted(test_stems),
        'val_regions': sorted(list(val_regions_set)),
        'test_regions': sorted(list(test_regions_set)),
        'train_regions': sorted([r for r in all_regions if r not in val_regions_set and r not in test_regions_set])
    }


class SpatialRegionKFold:
    """
    Spatial Region-Based K-Fold Cross-Validation Splitter.

    Ensures entire geographic sections or catchment blocks remain grouped together in either train or val,
    preventing spatial autocorrelation leakage across cross-validation folds.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, stems: List[str]):
        """
        Yields (train_indices, val_indices) for each fold.
        """
        # Map each sample index to its region group
        region_map: Dict[str, List[int]] = {}
        for idx, s in enumerate(stems):
            reg, _ = parse_sample_spatial_key(s)
            if reg not in region_map:
                region_map[reg] = []
            region_map[reg].append(idx)

        regions = sorted(list(region_map.keys()))
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(regions)

        # Distribute regions across folds
        folds: List[List[int]] = [[] for _ in range(self.n_splits)]
        for i, reg in enumerate(regions):
            fold_idx = i % self.n_splits
            folds[fold_idx].extend(region_map[reg])

        for f_idx in range(self.n_splits):
            val_idx = np.array(folds[f_idx])
            train_idx = np.array([idx for k, f in enumerate(folds) if k != f_idx for idx in f])
            yield train_idx, val_idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Spatial Data Leakage Audit & Region Splitter")
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='Path to data YAML or dataset root')
    parser.add_argument('--audit', action='store_true', default=True, help='Run spatial leakage audit')
    args = parser.parse_args()

    audit_spatial_splits(args.data, verbose=True)
