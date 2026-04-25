import numpy as np

def get_concentration_range():
    return np.geomspace(1e-10, 10, 500)

def compute_sft_profile(gamma_max, kl, cmc):
    R = 8314 # gas constant in mJ/(mol·K)
    T = 298.15 # temperature in K (25°C)
    gamma0 = 72.0 # surface tension of pure water in mN/m

    c_range = get_concentration_range()
    sft = []

    for c in c_range:
        if c < cmc:
            y = gamma0 - (R * T * gamma_max * np.log(1 + kl * c * 1000))
        else:
            y = gamma0 - (R * T * gamma_max * np.log(1 + kl * cmc * 1000))
        sft.append(y)

    log_c = np.log10(c_range)
    return log_c, np.array(sft)