def domain_shift_severity(stats_A, stats_B):
    """
    Computes a 0-1 severity score describing how different two datasets are,
    based on sampling rate, SNR, spectral centroid, and artifact rate.

    Parameters
    ----------
    stats_A : dict
        Summary statistics for dataset A. Expected keys: 'fs' (float, Hz),
        'snr' (float, dB), 'centroid' (float, Hz), 'artifact' (float, 0-1 rate).
    stats_B : dict
        Same structure as stats_A, for dataset B.

    Returns
    -------
    float
        Severity score in [0, 1]. 0 = identical domains, higher = more
        severe distribution shift. Computed as the mean of four normalized
        component differences (sampling rate, SNR, spectral centroid,
        artifact rate).

    Notes
    -----
    Normalization constants (30 for SNR, 20 for centroid) are empirical
    scaling factors. TODO: confirm exact rationale with Member 2.
    """
    rate_diff = abs(stats_A["fs"] - stats_B["fs"]) / max(stats_A["fs"], stats_B["fs"])
    snr_diff = abs(stats_A["snr"] - stats_B["snr"]) / 30
    centroid_diff = abs(stats_A["centroid"] - stats_B["centroid"]) / 20
    artifact_diff = abs(stats_A["artifact"] - stats_B["artifact"])
    return (rate_diff + snr_diff + centroid_diff + artifact_diff) / 4