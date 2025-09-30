import argparse
from .advanced_simulator import AdvancedRockfallSimulator, TerrainParams


def parse_args():
    parser = argparse.ArgumentParser(description='Run advanced rockfall simulator')
    parser.add_argument('--num-cells', type=int, default=20000, help='Number of terrain cells to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', type=str, default='advanced_rockfall_dataset.csv', help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    params = TerrainParams(num_cells=args.num_cells)
    sim = AdvancedRockfallSimulator(random_state=args.seed, terrain=params)
    df, meta = sim.generate()
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({df.shape[0]} rows, {df.shape[1]} cols)")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()


