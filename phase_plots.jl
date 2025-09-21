using LinearAlgebra
using Plots
using Colors

# ---------- helpers ----------
# 1σ ellipse points around μ with covariance Σ (2×2), k=1 ⇒ 1σ
function ellipse_points(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real};
                        k::Real = 1.0, npts::Int = 16)
    # Symmetrize & clamp tiny negatives from rounding
    Σs = Symmetric(0.5 .* (Σ .+ Σ'))
    vals, vecs = eigen(Σs)
    λ = max.(vals, 0.0)                # ensure nonnegative
    R = vecs
    s = k .* sqrt.(λ)                  # principal radii
    θ = range(0, 2π; length=npts)
    xs = similar(θ); ys = similar(θ)
    for (i, ang) in enumerate(θ)
        v = R * (s .* [cos(ang); sin(ang)])
        p = μ .+ v
        xs[i] = p[1]; ys[i] = p[2]
    end
    xs, ys
end

# Discrete step using 3rd-order Taylor for e^{A dt}; multiplicative noise covariance update
function simulate_mean_cov(A::Matrix, F::Matrix; μ0=[1.2, -0.8], Σ0=0.05I(2),
                           dt=0.01, T=8.0)
    I2  = Matrix(I, 2, 2)
    A2  = A*A
    A3  = A2*A
    dt2 = dt*dt
    dt3 = dt2*dt
    Ad  = I2 + dt*A + 0.5*dt2*A2 + (1/6)*dt3*A3

    ts = collect(0.0:dt:T)
    K  = length(ts)
    μ  = zeros(2, K);  μ[:,1] = μ0
    Σ  = zeros(2, 2, K); Σ[:,:,1] = Matrix(Σ0)

    for k in 1:K-1
        μnext = Ad * μ[:,k]
        Σprev = Σ[:,:,k]
        Σnext = Ad * Σprev * Ad' + dt * (F * (Σprev + μnext*μnext') * F')
        Σ[:,:,k+1] = 0.5 .* (Σnext .+ Σnext')   # symmetrize
        μ[:,k+1]   = μnext
    end
    ts, μ, Σ
end

# Build (A,F) from the parametrization for u ≡ 0, C = 0 (phase plot of the state)
# A = -1/2 P^{-1}(Q + F̃'F̃ + S),  F = L_{P^{-1}} F̃
function AF_from_params(Pinv::Matrix, Q::Matrix, Ftilde::Matrix, S::Matrix)
    LPinv = cholesky(Symmetric(Pinv)).L
    A = -0.5 .* (Pinv * (Q + Ftilde' * Ftilde + S))
    F = LPinv * Ftilde
    A, F
end
# ===== AAU style (your sizes) =====
const AAU_BLUE   = parse(Colorant, "#211A52")
const AAU_GREEN  = parse(Colorant, "#376C00")
const AAU_GOLD   = parse(Colorant, "#C9A227")
const AAU_G      = parse(Colorant, "#211A52")

default(
    guidefontcolor = AAU_BLUE,
    tickfontcolor  = AAU_BLUE,
    foreground_color_subplot = AAU_BLUE,
    background_color = nothing,
    grid = :y, gridalpha = 0.15, gridcolor = AAU_BLUE,
    tickfontsize   = 28,
    guidefontsize  = 30,
    legendfontsize = 26,
    titlefontsize  = 32,
)

# ===== Toggles (3): set any to false to hide globally =====
SHOW_GRID   = false
SHOW_TICKS  = true
SHOW_LABELS = false

# per-panel settings from toggles
grid_setting = SHOW_GRID ? :y : :off
xticks_set   = SHOW_TICKS ? :auto : false
yticks_set   = SHOW_TICKS ? :auto : false
xlabel_txt   = SHOW_LABELS ? "x₁" : ""
ylabel_txt   = SHOW_LABELS ? "x₂" : ""

# ===== Builders =====
function AF_from_params(Pinv::AbstractMatrix, Q::AbstractMatrix,
                        Ftilde::AbstractMatrix, S::AbstractMatrix)
    PinvM = Matrix(Pinv); QM = Matrix(Q)
    A = -0.5 .* (PinvM * (QM + Ftilde' * Ftilde + S))
    LPinv = cholesky(Symmetric(PinvM)).L
    F = LPinv * Ftilde
    return A, F
end

function ellipse_points(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real};
                        k::Real = 1.0, npts::Int = 16)
    Σs = Symmetric(0.5 .* (Σ .+ Σ'))
    vals, vecs = eigen(Σs)
    λ = max.(vals, 0.0)
    s = k .* sqrt.(λ)
    θ = range(0, 2π; length=npts)
    x = similar(θ); y = similar(θ)
    for (i, ang) in enumerate(θ)
        p = μ .+ vecs * (s .* [cos(ang); sin(ang)])
        x[i] = p[1]; y[i] = p[2]
    end
    x, y
end

function simulate_mean_cov(A::Matrix, F::Matrix; μ0=[1.2,-0.8], Σ0=0.02I(2),
                           dt=0.01, T=10.0)   # longer sim
    I2  = Matrix(I, 2, 2)
    A2, A3 = A*A, A*A*A
    Ad  = I2 + dt*A + 0.5*(dt^2)*A2 + (dt^3/6)*A3

    ts = collect(0.0:dt:T)
    K  = length(ts)
    μ  = zeros(2, K);  μ[:,1] = μ0
    Σ  = zeros(2, 2, K); Σ[:,:,1] = Matrix(Σ0)

    for k in 1:K-1
        μnext = Ad * μ[:,k]
        Σprev = Σ[:,:,k]
        Σnext = Ad * Σprev * Ad' + dt * (F * (Σprev + μnext*μnext') * F')
        Σ[:,:,k+1] = 0.5 .* (Σnext .+ Σnext')
        μ[:,k+1]   = μnext
    end
    ts, μ, Σ
end

# ===== Nominal params =====
n = 2
Pinv_nom = [1.0 0.2; 0.2 0.7]
Q_nom    = 0.6 .* I(n)
F̃_nom   = [0.22  0.00; -0.08  0.18]
S_nom    = [0.0 0.45; -0.45 0.0]

A_nom, F_nom = AF_from_params(Pinv_nom, Q_nom, F̃_nom, S_nom)
A_Q2,  F_Q2  = AF_from_params(Pinv_nom,  2.0 .* Q_nom,    F̃_nom,      S_nom)
A_S2,  F_S2  = AF_from_params(Pinv_nom,  Q_nom,           F̃_nom,      2.0 .* S_nom)
A_F2,  F_F2  = AF_from_params(Pinv_nom,  Q_nom,           2.0 .* F̃_nom, S_nom)
A_P2,  F_P2  = AF_from_params(0.5 .* Pinv_nom, Q_nom,     F̃_nom,      S_nom)  # P doubled ⇒ P⁻¹ halved

# ===== 8 initial conditions (two per quadrant) =====
μ0s = [
    [ 1.6,  1.2], [ 0.9,  0.6],
    [-1.5,  1.1], [-0.8,  0.5],
    [-1.6, -1.2], [-0.9, -0.6],
    [ 1.5, -1.1], [ 0.8, -0.5],
]

# Pre-simulate
function simulate_all(A, F, μ0s; dt=0.01, T=4.0)
    [simulate_mean_cov(A, F; μ0=μ0, Σ0=0.02I(2), dt=dt, T=T) for μ0 in μ0s]
end
bundles_nom = simulate_all(A_nom, F_nom, μ0s)
bundles_Q2  = simulate_all(A_Q2,  F_Q2,  μ0s)
bundles_S2  = simulate_all(A_S2,  F_S2,  μ0s)
bundles_F2  = simulate_all(A_F2,  F_F2,  μ0s)
bundles_P2  = simulate_all(A_P2,  F_P2,  μ0s)

# Axis limits (from all)
function global_limits(all_bundles; skip=20, kσ=1.0)
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for bundles in all_bundles
        for (ts, μ, Σ) in bundles
            for i in 1:skip:size(μ,2)
                xs, ys = ellipse_points(μ[:,i], Σ[:,:,i]; k=kσ, npts=16)
                xmin = min(xmin, minimum(xs), μ[1,i]); xmax = max(xmax, maximum(xs), μ[1,i])
                ymin = min(ymin, minimum(ys), μ[2,i]); ymax = max(ymax, maximum(ys), μ[2,i])
            end
        end
    end
    padx = 0.08*(xmax - xmin); pady = 0.08*(ymax - ymin)
    (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)
end
xlims_all, ylims_all = global_limits([bundles_nom, bundles_Q2, bundles_S2, bundles_F2, bundles_P2])

function draw_scenario!(plt, sp, title_str, bundles; bottom_xlabel_ids=(14, 16))
    use_xlabel = sp in bottom_xlabel_ids
    if use_xlabel
        plot!(plt; subplot=sp, aspect_ratio=:equal,
              xlims=xlims_all, ylims=ylims_all, grid=grid_setting,
              xticks=xticks_set, yticks=yticks_set,
              xlabel=title_str, ylabel=ylabel_txt, legend=false, title=nothing)
    else
        plot!(plt; subplot=sp, title=title_str, aspect_ratio=:equal,
              xlims=xlims_all, ylims=ylims_all, grid=grid_setting,
              xticks=xticks_set, yticks=yticks_set,
              xlabel=xlabel_txt, ylabel=ylabel_txt, legend=false)
    end

    styles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash, :dot, :solid]
    alphas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.85, 0.9, 0.95]
    for (k,(ts, μ, Σ)) in enumerate(bundles)
        ls, la = styles[k], alphas[k]
        plot!(plt, μ[1,:], μ[2,:]; subplot=sp, color=AAU_BLUE, lw=2, linestyle=ls, alpha=la, label="")
        scatter!(plt, [μ[1,1]], [μ[2,1]]; subplot=sp, marker=:star5, ms=7, color=AAU_GOLD, alpha=la, label="")
        idxs = 1:22:size(μ,2)
        L = length(collect(idxs))
        for (j,i) in enumerate(idxs)
            xs, ys = ellipse_points(μ[:,i], Σ[:,:,i]; k=1.0, npts=16)
            fa = 0.20 - 0.14*(j-1)/max(L-1,1)
            plot!(plt, xs, ys; subplot=sp, seriestype=:shape,
                  fillcolor=AAU_G, fillalpha=fa,
                  linecolor=AAU_G, linealpha=0.05, lw=0.5, label="")
        end
    end
end

# ---------- legend row + 5×3 grid (stars close) ----------
toprow_h = 0.10      # legend row
gap_h    = 0.06      # thin gap rows  (↓ to bring rows closer)
h_center = 0.40      # center row height
h_star   = (1 - (2*gap_h + h_center)) / 2  # -> 0.24 with the numbers above

w_gap    = 0.10      # thin center column (↓ to bring columns closer)
w_side   = (1 - w_gap) / 2                 # -> 0.45

lay = @layout [
    a{0.1h};
    grid(5, 3;
         heights = [h_star, gap_h, h_center, gap_h, h_star],
         widths  = [w_side, w_gap, w_side])
]

plt = plot(layout=lay, size=(1700,1400))

# legend in subplot 1
plot!(plt; subplot=1, framestyle=:none, grid=false,
      xaxis=false, yaxis=false, xticks=false, yticks=false,
      xlim=(0,1), ylim=(1,2), legend=:bottom)
xleg = [0.0, 1.0]; yoff = 3.0; yleg = fill(yoff, 2)
plot!(plt, xleg, yleg; subplot=1, color=AAU_BLUE, lw=2, label="mean")
plot!(plt, xleg, yleg; subplot=1, seriestype=:shape, color=AAU_G, fillalpha=0.18, label="1σ ellipse")
scatter!(plt, [0.5], [yoff]; subplot=1, marker=:star5, ms=7, color=AAU_GOLD, label="start")
plot!(plt; subplot=1, legendcolumns=3,
      foreground_color_legend=nothing, background_color_legend=nothing)

# blank all non-star cells inside the 5×3 grid
for sp in (3,5,6,7,11,12,13,15)  # everything except 2,4,9,14,16
    plot!(plt; subplot=sp, framestyle=:none, grid=false,
          xaxis=false, yaxis=false, xticks=false, yticks=false)
end

# place scenarios at the five stars:
# row-major indices after the legend:
# row1: 2,3,4  | row2: 5,6,7 | row3: 8,9,10 | row4: 11,12,13 | row5: 14,15,16
draw_scenario!(plt, 2,  "Q doubled",            bundles_Q2)   # top-left
draw_scenario!(plt, 4,  "S doubled",            bundles_S2)   # top-right
draw_scenario!(plt, 9,  "Nominal",              bundles_nom)  # center
draw_scenario!(plt, 14, "P doubled",            bundles_P2)   # bottom-left (xlabel)
draw_scenario!(plt, 16, L"$\tilde{F}$ doubled", bundles_F2)   # bottom-right (xlabel)

display(plt)
savefig(plt, "phase_star_8ICs.tikz")
# savefig(plt, "phase_star_8ICs.png")
# using PGFPlotsX; pgfplotsx(); savefig(plt, "phase_star_8ICs.tikz")
