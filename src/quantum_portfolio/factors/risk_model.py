def factor_covariance(exposures, factor_cov, specific_var):
    cov = exposures @ factor_cov.reindex(index=exposures.columns, columns=exposures.columns) @ exposures.T
    for a,v in specific_var.items(): cov.loc[a,a] += v
    return cov
