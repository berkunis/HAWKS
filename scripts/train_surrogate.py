"""Train the TrustSurrogate MLP on synthetic interaction data.

Pipeline:
  SyntheticDataGenerator.generate()
  → group by operator_id, shift trust by +1 to get T_{t+1}
  → 80/20 train/val split
  → Adam + MSE for ≤200 epochs
  → save to models/surrogate.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hawks.data.synthetic import SyntheticDataGenerator
from hawks.models.surrogate import TrustSurrogate

FEATURE_COLS = ["structural_risk", "ai_confidence", "ai_correctness", "operator_trust"]


def build_dataset(
    n_operators: int = 20, n_steps: int = 200, master_seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data and construct (features, T_{t+1}) pairs."""
    gen = SyntheticDataGenerator(
        n_operators=n_operators, n_steps=n_steps, master_seed=master_seed
    )
    df = gen.generate()

    # Within each operator, the next row's trust is T_{t+1}.
    df = df.sort_values(["operator_id", "step"]).reset_index(drop=True)
    df["trust_next"] = df.groupby("operator_id")["operator_trust"].shift(-1)

    # Drop the last step per operator (no target).
    df = df.dropna(subset=["trust_next"]).reset_index(drop=True)

    features = torch.tensor(
        df[FEATURE_COLS].astype("float32").values, dtype=torch.float32
    )
    targets = torch.tensor(
        df["trust_next"].astype("float32").values, dtype=torch.float32
    )
    return features, targets


def main() -> None:
    print("Generating synthetic data...")
    features, targets = build_dataset()
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")

    # 80/20 split
    n = features.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(features[train_idx], targets[train_idx])
    val_ds = TensorDataset(features[val_idx], targets[val_idx])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = TrustSurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("Training...")
    for epoch in range(1, 201):
        # Train
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
            train_n += xb.shape[0]

        if epoch % 20 == 0 or epoch == 1:
            # Validate
            model.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * xb.shape[0]
                    val_n += xb.shape[0]

            print(
                f"  epoch {epoch:3d}  "
                f"train_mse={train_loss / train_n:.6f}  "
                f"val_mse={val_loss / val_n:.6f}"
            )

    # Final metrics
    model.eval()
    with torch.no_grad():
        train_pred = model(features[train_idx])
        final_train = criterion(train_pred, targets[train_idx]).item()
        val_pred = model(features[val_idx])
        final_val = criterion(val_pred, targets[val_idx]).item()

    print(f"\nFinal  train_mse={final_train:.6f}  val_mse={final_val:.6f}")

    # Save
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "surrogate.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
