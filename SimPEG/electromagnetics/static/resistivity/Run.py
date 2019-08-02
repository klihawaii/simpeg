import numpy as np

from SimPEG import (
    maps, optimization, inversion, inverse_problem, directives,
    data_misfit, regularization
)


def run_inversion(
    m0, survey, actind, mesh,
    std, eps,
    maxIter=15, beta0_ratio=1e0,
    coolingFactor=5, coolingRate=2,
    upper=np.inf, lower=-np.inf,
    use_sensitivity_weight=True,
    alpha_s=1e-4,
    alpha_x=1.,
    alpha_y=1.,
    alpha_z=1.,
):
    """
    Run DC inversion
    """
    dmisfit = data_misfit.L2DataMisfit(survey)
    uncert = abs(survey.dobs) * std + eps
    dmisfit.W = 1./uncert
    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(actind.sum()))
    # Related to inversion
    if use_sensitivity_weight:
        reg = regularization.Sparse(mesh, indActive=actind, mapping=regmap)
        reg.alpha_s = alpha_s
        reg.alpha_x = alpha_x
        reg.alpha_y = alpha_y
        reg.alpha_z = alpha_z
    else:
        reg = regularization.Tikhonov(mesh, indActive=actind, mapping=regmap)
        reg.alpha_s = alpha_s
        reg.alpha_x = alpha_x
        reg.alpha_y = alpha_y
        reg.alpha_z = alpha_z

    opt = optimization.ProjectedGNCG(maxIter=maxIter, upper=upper, lower=lower)
    invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    target = directives.TargetMisfit()
    # Need to have basice saving function
    update_Jacobi = directives.UpdatePreconditioner()
    if use_sensitivity_weight:
        updateSensW = directives.UpdateSensitivityWeights()
        directiveList = [
            beta, betaest, target, updateSensW, update_Jacobi
        ]
    else:
        directiveList = [
            beta, betaest, target, update_Jacobi
        ]
    inv = inversion.BaseInversion(
        invProb, directiveList=directiveList
        )
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run inversion
    mopt = inv.run(m0)
    return mopt, invProb.dpred

