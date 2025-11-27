import json
from pathlib import Path

from src.stats import stereoset_stats, crows_pairs_stats, sst2_stats

def main():
    print("=== StereoSet stats ===")
    ss_stats = stereoset_stats()
    print(json.dumps(ss_stats, indent=2, sort_keys=True))

    print("\n=== CrowS-Pairs stats ===")
    cp_stats = crows_pairs_stats()
    print(json.dumps(cp_stats, indent=2, sort_keys=True))

    print("\n=== SST-2 (GLUE) stats ===")
    sst_stats = sst2_stats()
    print(json.dumps(sst_stats, indent=2, sort_keys=True))

    # dump to a JSON file
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stereoset": ss_stats,
                "crows_pairs": cp_stats,
                "sst2": sst_stats,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"\nSaved stats to {out_dir / 'dataset_stats.json'}")

if __name__ == "__main__":
    main()
