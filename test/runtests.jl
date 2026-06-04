using EPEDNN
using Test

# Pedestal HEIGHT (MPa, omega-star "H" diamagnetic model, sol0) for the legacy 10-input net.
legacy_H_height(m, p) = m(p.a, p.betan, p.bt, p.delta, p.ip, p.kappa, p.m, p.neped, p.r, p.zeffped;
    warn_nn_train_bounds=false).pressure.H.H

# Ensemble height (MPa) at the legacy old-data defaults (nesep_ratio=0.25, tesep=75 eV).
ensemble_height(ens, p) = EPEDNN.ensemble_predict(
    ens, [p.a, p.betan, p.bt, p.delta, p.ip, p.kappa, p.m, p.neped, 0.25, p.r, 75.0, p.zeffped]).mean

@testset "EPEDNN.jl" begin

    legacy = EPEDNN.loadmodelonce("EPED1NNmodel.bson")
    ens = EPEDNN.loadensemble()

    @testset "ensemble matches deployed server (ITER 15MA)" begin
        # The EPED explorer/server returns 78.02 kPa +/- 0.52 for the ITER 15MA base point.
        x = [2.0, 2.0, 5.3, 0.49, 15.0, 1.85, 2.5, 7.0, 0.25, 6.2, 75.0, 1.5]
        u = EPEDNN.ensemble_uncertainty(ens, x)
        @test isapprox(u.height * 1000, 78.02; atol=0.5)   # kPa
        @test isapprox(u.sigma * 1000, 0.52; atol=0.2)     # kPa
        @test u.extrapolation == 0.0
        @test u.in_distribution
    end

    @testset "legacy net and ensemble agree in-distribution (H, sol0)" begin
        # Conventional-tokamak points, in the comfortable range of BOTH nets (not ITER-scale,
        # where the new scans were added precisely because the legacy net extrapolates).
        # (a, betan, bt, delta, ip[MA], kappa, m, neped[1e19], r, zeffped)
        points = [
            (a=0.6, betan=2.0, bt=2.0, delta=0.30, ip=1.5, kappa=1.8, m=2.0, neped=4.0, r=1.70, zeffped=1.8),  # DIII-D-like
            (a=0.9, betan=2.0, bt=2.5, delta=0.30, ip=2.5, kappa=1.7, m=2.0, neped=5.0, r=2.90, zeffped=1.8),  # JET-like
            (a=0.5, betan=1.8, bt=2.1, delta=0.25, ip=1.2, kappa=1.7, m=2.0, neped=5.0, r=1.65, zeffped=2.0),
        ]
        for p in points
            h_legacy = legacy_H_height(legacy, p)
            h_ens = ensemble_height(ens, p)
            rel = abs(h_ens - h_legacy) / h_legacy
            @test rel < 0.20
        end
    end

end
