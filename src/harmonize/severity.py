def domain_shift_severity(stats_A, stats_B):
    """
    Computes a 0-1 severity score describing how different two datasets are,
    based on sampling rate, SNR, spectral centroid, and artifact rate.
    """
    rate_diff = abs(stats_A["fs"] - stats_B["fs"]) / max(stats_A["fs"], stats_B["fs"])
    snr_diff = abs(stats_A["snr"] - stats_B["snr"]) / 30
    centroid_diff = abs(stats_A["centroid"] - stats_B["centroid"]) / 20
    artifact_diff = abs(stats_A["artifact"] - stats_B["artifact"])
    return (rate_diff + snr_diff + centroid_diff + artifact_diff) / 4
