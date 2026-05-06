import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["temperature", "rpm", "load", "vibration"]


def generate_synthetic_failure_data(n_samples: int = 8200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic sensor data with realistic failure conditions."""
    rng = np.random.default_rng(random_state)

    temperature = rng.normal(loc=110, scale=30, size=n_samples)
    temperature = np.clip(temperature, 20, 250)

    rpm = rng.normal(loc=3200, scale=1200, size=n_samples)
    rpm = np.clip(rpm, 500, 8000)

    load = rng.normal(loc=45, scale=25, size=n_samples)
    load = np.clip(load, 0, 100)

    vibration = rng.normal(loc=3.0, scale=2.0, size=n_samples)
    vibration = np.clip(vibration, 0, 10)

    # Build a failure score using a weighted combination of sensor readings.
    failure_score = (
        0.40 * (temperature / 250)
        + 0.30 * (load / 100)
        + 0.20 * (vibration / 10)
        + 0.10 * (rpm / 8000)
    )

    # Add moderate noise to reflect real sensor uncertainty.
    failure_score += rng.normal(scale=0.05, size=n_samples)

    # Use a clear threshold for more consistent failure / healthy separation.
    threshold = 0.55
    failure = (failure_score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "temperature": np.round(temperature, 1),
            "rpm": np.round(rpm, 1),
            "load": np.round(load, 1),
            "vibration": np.round(vibration, 2),
            "failure": failure,
        }
    )

    # Add noise and sensor drift to reflect realistic conditions.
    df["temperature"] += rng.normal(scale=0.4, size=n_samples)
    df["rpm"] += rng.normal(scale=20, size=n_samples)
    df["load"] += rng.normal(scale=1.5, size=n_samples)
    df["vibration"] += rng.normal(scale=0.05, size=n_samples)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].clip(lower=[20, 500, 0, 0], upper=[250, 8000, 100, 10])

    return df


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data quality cleaning on raw sensor inputs."""
    if not set(FEATURE_COLUMNS + ["failure"]).issubset(df.columns):
        raise ValueError("DataFrame missing required sensor columns.")

    df = df.copy()
    df = df.dropna()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df = df[(df.temperature >= 20) & (df.temperature <= 250)]
    df = df[(df.rpm >= 500) & (df.rpm <= 8000)]
    df = df[(df.load >= 0) & (df.load <= 100)]
    df = df[(df.vibration >= 0) & (df.vibration <= 10)]

    return df.reset_index(drop=True)


def create_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from cleaned dataset."""
    return df[FEATURE_COLUMNS].values


def create_target_vector(df: pd.DataFrame) -> np.ndarray:
    """Extract target failure labels."""
    return df["failure"].values
