import warnings

import torch


PIPELINE_VERSION = 6


def save_checkpoint(path, state_dict, algorithm, state_size, window_size):
    torch.save(
        {
            "pipeline_version": PIPELINE_VERSION,
            "algorithm": algorithm,
            "state_size": state_size,
            "window_size": window_size,
            "state_dict": state_dict,
        },
        path,
    )


def load_checkpoint(path, map_location, algorithm, allow_legacy=True, return_metadata=False):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        if payload.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError(
                f"checkpoint pipeline version {payload.get('pipeline_version')} is incompatible "
                f"with version {PIPELINE_VERSION}"
            )
        if payload.get("algorithm") != algorithm:
            raise ValueError(f"expected a {algorithm} checkpoint, got {payload.get('algorithm')}")
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
        result = payload["state_dict"]
        return (result, metadata) if return_metadata else result

    if not allow_legacy:
        raise ValueError(
            "legacy checkpoint predates the corrected environment; retrain it or explicitly allow legacy weights"
        )
    warnings.warn(
        "Loading a legacy checkpoint trained with the old environment timing. Retrain before comparing results.",
        RuntimeWarning,
        stacklevel=2,
    )
    metadata = {"pipeline_version": "legacy", "algorithm": algorithm}
    return (payload, metadata) if return_metadata else payload
