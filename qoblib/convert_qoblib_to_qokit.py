import pandas as pd
import numpy as np
import gzip
import argparse

def load_clean_df(filepath, column_names):
    with gzip.open(filepath, 'rt') as f:
        df = pd.read_csv(f, sep=r"\s+", comment='#', names=column_names, engine="python")
    return df

def main(price_path, cov_path, output_csv, lambda_, day=0):
    # Load data
    price_df = load_clean_df(price_path, ["day", "symbol", "price"])
    cov_df = load_clean_df(cov_path, ["day", "sym1", "sym2", "cov"])

    # Filter for the selected day
    df_price_day = price_df[price_df["day"] == day]
    df_cov_day = cov_df[cov_df["day"] == day]

    # Consistent symbol order
    symbols = sorted(df_price_day["symbol"].unique())
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    N = len(symbols)

    # Create price vector μ
    μ = df_price_day.set_index("symbol").loc[symbols]["price"].values
    μ_normalized = μ / np.mean(μ)

    # Build covariance matrix Σ
    # Build covariance matrix Σ, skipping any unknown tickers
    Σ = np.zeros((N, N))
    for _, row in df_cov_day.iterrows():
        s1, s2 = row["sym1"], row["sym2"]
        if s1 not in symbol_to_idx or s2 not in symbol_to_idx:
            # optionally log: print(f"[warning] skipping cov for {s1},{s2}")
            continue
        i, j = symbol_to_idx[s1], symbol_to_idx[s2]
        Σ[i, j] = row["cov"]
    Σ = (Σ + Σ.T) / 2  # Symmetrize


    # Compute cost matrix: C = Σ - λμμᵀ
    cost_matrix = Σ - lambda_ * np.outer(μ_normalized, μ_normalized)

    # Save to CSV
    np.savetxt(output_csv, cost_matrix, delimiter=",")
    print(f"✅ Saved cost matrix for day {day} to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QOBLIB S&P data to QOKit cost matrix")
    parser.add_argument("--prices", required=True, help="Path to stock_prices.txt.gz")
    parser.add_argument("--covs", required=True, help="Path to covariance_matrices.txt.gz")
    parser.add_argument("--out", default="cost_matrix.csv", help="Output CSV filename")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5, help="Risk aversion parameter (λ)")
    parser.add_argument("--day", type=int, default=0, help="Day index to extract")

    args = parser.parse_args()
    main(args.prices, args.covs, args.out, args.lambda_, args.day)