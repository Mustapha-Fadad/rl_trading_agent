def split_validation_windows(data, observation_window, num_windows, window_rows):
    """Reserve consecutive validation regimes while retaining lookback context."""
    reserve_rows = num_windows * window_rows + observation_window
    if len(data) <= reserve_rows + observation_window:
        raise ValueError("not enough training data for the configured validation windows")

    train_only = data.iloc[:-reserve_rows].copy()
    reserve = data.iloc[-reserve_rows:].copy()
    windows = []
    for index in range(num_windows):
        start = index * window_rows
        stop = start + window_rows + observation_window
        window = reserve.iloc[start:stop].copy()
        window.attrs = data.attrs.copy()
        windows.append(window)

    train_only.attrs = data.attrs.copy()
    return train_only, windows
