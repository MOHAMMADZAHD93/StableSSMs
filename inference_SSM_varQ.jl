using Turing, ReverseDiff, LinearAlgebra, Distributions, Random, StatsPlots
using JLD2

function skew_matrix(v::AbstractVector, n::Int)
    T = eltype(v)
    S = zeros(T, n, n)
    idx = 1
    for i in 1:(n-1)
        for j in (i+1):n
            S[i,j] = v[idx]
            S[j,i] = -v[idx]
            idx += 1
        end
    end
    return S
end

# ----------------------------
# Set seed and dimensions
# ----------------------------
Random.seed!(12345)
n = 4        # state dimension
q = 1        # observation dimension
ℓ = 2        # input dimension
m = 10        # number of trajectories
times = collect(0.0:0.0025:0.5)  # observation times

# ----------------------------
# Define constant scale matrices and their Cholesky factors
# ----------------------------
Σ_p = Matrix(I, n, n)
Σ_q = 2*Matrix(I, n, n)
L_Sigma_p = cholesky(Σ_p).L
L_Sigma_q = cholesky(Σ_q).L

# ----------------------------
# Ground truth hyperparameters 
# ----------------------------
k_p = n + 4
k_q = n + 4
μ_s = zeros(div(n*(n-1),2))
Σ_s = 0.01*Matrix(I, div(n*(n-1),2), div(n*(n-1),2))
μ_f = zeros(n*n)
Σ_f = 2*Matrix(I, n*n, n*n)
μ_c = zeros(q*n)
Σ_c = 2*Matrix(I, q*n, q*n)
μ_z = zeros((n+q+n)*ℓ)
Σ_z = 0.1*Matrix(I, (n+q+n)*ℓ, (n+q+n)*ℓ)
Σ_z[1:n, 1:n] = 0.1*Matrix(I, n, n)
ε = 1e-4
dim_u = (n+q+n)
μ_u = zeros(div(dim_u*(dim_u-1),2))
Σ_u = 1.0*Matrix(I, div(dim_u*(dim_u-1),2), div(dim_u*(dim_u-1),2))
dim_v = ℓ
μ_v = zeros(div(dim_v*(dim_v-1),2))
Σ_v = 1.0*Matrix(I, div(dim_v*(dim_v-1),2), div(dim_v*(dim_v-1),2))
alpha_rho = 2.0
beta_rho = 2.0
k_γ = 1.0
θ_γ = 5.0
σ_shape = 1.0
σ_scale = 0.5

# ----------------------------
# Generate ground truth parameters (using the same parametrization)
# ----------------------------

# L_Pinv (for P⁻¹ ~ Wishart(k_p, Σ_p))
L_Pinv = zeros(n, n)
for i in 1:n
    L_Pinv[i, i] = sqrt(rand(Chisq(k_p - i + 1)))
    for j in 1:(i-1)
        L_Pinv[i, j] = rand(Normal(0, 1))
    end
end
L_Pinv = L_Sigma_p* L_Pinv 

# L_Q (for Q ~ Wishart(k_q, Σ_q))
L_Q = zeros(n, n)
for i in 1:n
    L_Q[i, i] = sqrt(rand(Chisq(k_q - i + 1)))
    for j in 1:(i-1)
        L_Q[i, j] = rand(Normal(0, 1))
    end
end
L_Q = L_Sigma_q * L_Q 

# Skew-symmetric matrix S
s_veck = rand(MvNormal(μ_s, Σ_s))
S = skew_matrix(s_veck, n)

# F̃ matrix
vec_F̃ = rand(MvNormal(μ_f, Σ_f))
F̃ = reshape(vec_F̃, n, n)

# C matrix
vec_C = rand(MvNormal(μ_c, Σ_c))
C = reshape(vec_C, q, n)

# Z matrix ([B̃; D; G̃] / γ)
# ----------------------------
# For U factor:
dim_u = n + q + n  # 5 in this case
s_u = rand(MvNormal(μ_u, Σ_u))
S_u = skew_matrix(s_u, dim_u)
I_dim = Matrix(I, dim_u, dim_u)
# Sample E_u (a diagonal matrix with entries ±1)
E_u = Diagonal(rand([-1, 1], dim_u))
U_full = E_u * ((I_dim - S_u)/(I_dim + S_u))  # full 5×5 orthogonal matrix
U_z = U_full[:, 1:ℓ]                        # take first ℓ=2 columns

# For V factor:
dim_v = ℓ
s_v = rand(MvNormal(μ_v, Σ_v))
S_v = skew_matrix(s_v, dim_v)
I2 = Matrix(I, dim_v, dim_v)
# Sample E_v (a diagonal matrix with entries ±1)
E_v = Diagonal(rand([-1, 1], dim_v))
V_z = E_v * ((I2 - S_v)/(I2 + S_v))
# Sample ℓ singular values (with the first fixed near 1)
sigma_vals = [rand(Uniform(0, 1 - ε)) for i in 1:ℓ-1]
sigma = Diagonal(vcat(1-ε,sigma_vals))

# Construct Z = U_z * diag(σ) * V_zᵀ (a 5×2 matrix)
Z = U_z * sigma * transpose(V_z)

γ = 3.0
B̃ = γ .* Z[1:n, :]
D   = γ .* Z[n+1:n+q, :]
G̃  = γ .* Z[n+q+1:end, :]

# ρ (ground truth)
ρ = 0.3

# Measurement noise standard deviation
σ = 0.15 

# ----------------------------
# Construct system matrices 
# ----------------------------
P_inv = L_Pinv * L_Pinv'
P = inv(P_inv)
Q_mat = L_Q * L_Q'
A = -0.5 * P_inv * (Q_mat + F̃' * F̃ + C' * C + S)
F = L_Pinv * F̃
G = L_Pinv * G̃
B = P_inv * (L_Q * B̃ - ρ * F̃' * G̃ - C' * D)

# --- Define a 2-input function ---
N_term = 6  # number of Fourier terms
# For each of the 2 inputs, generate its own set of Fourier coefficients.
fn_array = [randn(1 + 2*N_term) for i in 1:ℓ]
function create_u_func_vec(t, fn_array, N)
    u = zeros(length(fn_array))
    f = 1/0.2
    for j in 1:length(fn_array)
        result = fn_array[j][1]
        for k in 1:N
            result += fn_array[j][k+1]   * cos(2*pi*k*j*f*t) +
                      fn_array[j][k+N+1] * sin(2*pi*k*j*f*t)
        end
        u[j] = result
    end
    return u
end
u_func(t) = create_u_func_vec(t, fn_array, N_term)

# --- Simulate m trajectories from the discrete SSM ---
data = Vector{Matrix{Float64}}(undef, m)
inputs = fill(u_func, m)  # use the same u_func for all trajectories

for traj in 1:m
    x = zeros(n, length(times))
    y = zeros(q, length(times))
    x[:, 1] = zeros(n)
    for k in 1:(length(times)-1)
        dt = times[k+1] - times[k]
        u_val = u_func(times[k])  # u_val is now a 2-element vector
        # Noise covariance: dt*[1 ρ_true; ρ_true 1]
        Q_noise = dt * [1.0 ρ; ρ 1.0]
        w = rand(MvNormal(zeros(2), Q_noise))
        x[:, k+1] = x[:, k] + dt*A*x[:, k] + dt*B*u_val +
                    F*x[:, k]*w[1] + G*u_val*w[2]
    end
    for k in 1:length(times)
        u_val = u_func(times[k])
        y[:, k] = C*x[:, k] + D*u_val + rand(MvNormal(zeros(q), (σ^2)*I(q)))
    end
    data[traj] = y
end

# plot one trajectory (e.g. observations from the first trajectory)
plot(times, vec(data[1][1, :]), xlabel="Time", ylabel="Observation", label="Trajectory 1", lw=2)

rand(Uniform(0, 1))

# ----------------------------
# Define the main model with Euler–Maruyama discretization.
# ----------------------------
@model function ssm_bayes_inference(data, inputs, times, m, n, ℓ, q, L_Sigma_p, L_Sigma_q)	    
    k_p = n + 2
    k_q = n + 2
    μ_s = zeros(div(n*(n-1),2))
    Σ_s = I(div(n*(n-1),2))
    μ_f = zeros(n*n)
    Σ_f = I(n*n)
    μ_c = zeros(q*n)
    Σ_c = I(q*n)
    μ_z = zeros((n+q+n)*ℓ)
    Σ_z = I(n+q+n)
    alpha_rho = 2.0
    beta_rho = 2.0
    k_γ = 1.0
    θ_γ = 1.0
    σ_shape = 1.0
    σ_scale = 1.0
    ε = 1e-4
    
    dim_u = (n+q+n)
    μ_u = zeros(div(dim_u*(dim_u-1),2))
    Σ_u = 1.0*Matrix(I, div(dim_u*(dim_u-1),2), div(dim_u*(dim_u-1),2))
    dim_v = ℓ
    μ_v = zeros(div(dim_v*(dim_v-1),2))
    Σ_v = 1.0*Matrix(I, div(dim_v*(dim_v-1),2), div(dim_v*(dim_v-1),2))
    
    #### ρ ####
    ρ ~ Uniform(-0.95, 0.95)

    #### Inference on L_Pinv for P⁻¹ ~ Wishart(k_p, Σ_p) ####
    diag_val_p ~ arraydist([Chisq(k_p - i + 1) for i in 1:n])
    L_Pinv = zeros(eltype(ρ), n, n)
    for i in 1:n
        L_Pinv[i, i] = sqrt(diag_val_p[i])
        for j in 1:(i-1)
            L_Pinv[i, j] ~ Normal(0, 1)
        end
    end
    L_Pinv = L_Sigma_p * L_Pinv 

    #### Inference on L_Q for Q ~ Wishart(k_q, Σ_q) ####
    diag_val_q ~ arraydist([Chisq(k_p - i + 1) for i in 1:n])
    L_Q = zeros(eltype(ρ), n, n)
    for i in 1:n
        L_Q[i, i] = sqrt(diag_val_q[i])
        for j in 1:(i-1)
            L_Q[i, j] ~ Normal(0, 1)
        end
    end
    L_Q = L_Sigma_q * L_Q 
    
    #### Skew-symmetric matrix S ####
    s_veck ~ MvNormal(μ_s, Σ_s)
    S = skew_matrix(s_veck, n)
    
    #### F̃ matrix ####
    vec_F̃ ~ MvNormal(μ_f, Σ_f)
    F̃ = reshape(vec_F̃, n, n)
    
    #### C matrix ####
    vec_C ~ MvNormal(μ_c, Σ_c)
    C = Matrix(reshape(vec_C, q, n))
    
    #### Z matrix ([B̃; D; G̃] / γ) ####
    s_u ~ MvNormal(μ_u, Σ_u)
    S_u = skew_matrix(s_u, dim_u)
    s_v ~ MvNormal(μ_v, Σ_v)
    S_v = skew_matrix(s_v, dim_v)
    
    # --- Sample the sign diagonal matrices for U_z and V_z ---
    E_u_vec ~ filldist(Bernoulli(0.5), dim_u)
    # Map {0,1} to {-1, +1}
    E_u = Diagonal(E_u_vec .- 1)
    E_v_vec ~ filldist(Bernoulli(0.5), dim_v)
    E_v = Diagonal(E_v_vec .- 1)
    
    I_dim_u = Matrix(I, dim_u, dim_u)
    U_full = E_u * ((I_dim_u - S_u) / (I_dim_u + S_u))
    U_z = U_full[:, 1:ℓ]
    
    I_dim_v = Matrix(I, dim_v, dim_v)
    V_full = E_v * ((I_dim_v - S_v) / (I_dim_v + S_v))
    V_z = V_full  # V_z is a full dim_v×dim_v orthogonal matrix
    
    sigma_vals ~ arraydist([Uniform(0, 1 - ε) for i in 1:ℓ-1])
    Sigma_diag = Diagonal(vcat(1 - ε, sigma_vals))
    
    Z = U_z * Sigma_diag * transpose(V_z)
    
    γ ~ Gamma(k_γ, θ_γ)
    B̃ = γ .* Z[1:n, :]
    D   = γ .* Z[n+1:n+q, :]
    G̃  = γ .* Z[n+q+1:end, :]
    
    #### Measurement noise ####
    σ ~ InverseGamma(σ_shape, σ_scale)
    
    #### Construct system matrices ####
    P_inv = L_Pinv * transpose(L_Pinv)
    Q_mat = L_Q * transpose(L_Q)
    A = -0.5 * P_inv * (Q_mat + transpose(F̃) * F̃ + transpose(C) * C + S)
    A = Matrix(A)
    F = Matrix(L_Pinv * F̃)
    G = Matrix(L_Pinv * G̃)
    B = Matrix(P_inv * (L_Q * B̃ - ρ * transpose(F̃) * G̃ - transpose(C) * D))
    
    I_n = I(n)
    I_q = I(q)
    dt = times[2] - times[1]
    Adis = I(n)+dt*A+dt^2/2*A*A + dt^3/6*A*A*A
    Bdis = dt*B + dt^2/2*A*B + dt^3/6*A*A*B
    for traj in 1:m
        u_traj = inputs[traj]
        μ = zeros(n,1)
        Σ = zeros(n, n)
        # Loop over time steps
        for i in 1:(length(times) - 1)
            dt = times[i+1] - times[i]
            u_val = vec(u_traj(times[i]))
            μ = Adis*μ +   Bdis * u_val
            Σ = (Adis) * Σ * transpose(Adis) + dt * (F * (Σ + μ * transpose(μ)) * transpose(F) + G * (u_val * transpose(u_val)) * transpose(G) +
                 ρ * (F * μ * transpose(u_val) * transpose(G) + G * u_val * transpose(μ) * transpose(F)))
            Σ = 0.5 * (Σ + transpose(Σ))
            y_mean = C * μ + D * u_val
            y_cov = C * Σ * transpose(C) + (σ^2+1e-4) * I_q 
            data[traj][:, i] ~ MvNormal(vec(y_mean), y_cov)
        end
    end
end

using Optim 
L_Sigma_q = L_Sigma_q/2
model = ssm_bayes_inference(data, inputs, times, m, n, ℓ, q, L_Sigma_p, L_Sigma_q)
MAP_estimate = optimize(model, MAP(),  LBFGS(),Optim.Options(iterations=1_000, allow_f_increases=true))
chain = sample(model, HMC(0.01, 3; adtype=AutoReverseDiff(true)),init = MAP_estimate.values.array, 20000) 

save("chain_20000_HMC_4statesQ_new10M.jld2", "chain", chain)
#chain = load("chain_20000_HMC_4statesQ_new.jld2", "chain")

# ----------------------------
# Posterior predictive simulation from the chain
# ----------------------------
num_samples = size(chain, 1)
num_traj_chain = 100  # Trajectories to simulate per chain sample

times_extended = collect(0.0:0.0025:2)
num_traj_true = 100  # number of new trajectories for computing true statistics
u_vals_extended = [u_func(t) for t in times_extended]

posterior_means = zeros(q, length(times_extended), num_samples)
posterior_vars = zeros(q, length(times_extended), num_samples)

# Loop over (a subset of) posterior samples
for sample_idx in 19000:num_samples
    # --- Extract L_Pinv parameters ---
    diag_val_p = [chain[sample_idx][Symbol("diag_val_p[$i]")].data[1] for i in 1:n]
    L_Pinv_sample = zeros(n, n)
    for i in 1:n
        L_Pinv_sample[i, i] = sqrt(diag_val_p[i])
        for j in 1:(i-1)
            L_Pinv_sample[i, j] = chain[sample_idx][Symbol("L_Pinv[$i, $j]")].data[1]
        end
    end
    L_Pinv_sample = L_Sigma_p * L_Pinv_sample
    
    # --- Extract L_Q parameters ---
    diag_val_q = [chain[sample_idx][Symbol("diag_val_q[$i]")].data[1] for i in 1:n]
    L_Q_sample = zeros(n, n)
    for i in 1:n
        L_Q_sample[i, i] = sqrt(diag_val_q[i])
        for j in 1:(i-1)
            L_Q_sample[i, j] = chain[sample_idx][Symbol("L_Q[$i, $j]")].data[1]
        end
    end
    L_Q_sample = L_Sigma_q * L_Q_sample

    # --- Extract S (via its vector s_veck) ---
    s_veck_sample = [chain[sample_idx][Symbol("s_veck[$i]")].data[1] for i in 1:div(n*(n-1),2)]
    
    # --- Extract F̃ and C ---
    vec_F̃_sample = [chain[sample_idx][Symbol("vec_F̃[$i]")].data[1] for i in 1:(n*n)]
    F̃_sample = reshape(vec_F̃_sample, n, n)
    
    vec_C_sample = [chain[sample_idx][Symbol("vec_C[$i]")].data[1] for i in 1:(q*n)]
    C_sample = reshape(vec_C_sample, q, n)
    
    # --- Extract the Cayley transform parameters for [B̃; D; G̃] ---
    # Set dimensions: dim_u = n+q+n (here 5) and dim_v = ℓ (here 2)
    dim_u = n + q + n
    num_u = div(dim_u*(dim_u-1), 2)
    u_veck_sample = [chain[sample_idx][Symbol("s_u[$i]")].data[1] for i in 1:num_u]
    S_u_sample = skew_matrix(u_veck_sample, dim_u)
    
    dim_v = ℓ
    num_v = div(dim_v*(dim_v-1), 2)
    if num_v > 0
        v_veck_sample = [chain[sample_idx][Symbol("s_v[$i]")].data[1] for i in 1:num_v]
        S_v_sample = skew_matrix(v_veck_sample, dim_v)
    else
        S_v_sample = Matrix(I, dim_v, dim_v)
    end
    
    # --- Extract sign vectors and build sign matrices ---
    E_u_vec_sample = [chain[sample_idx][Symbol("E_u_vec[$i]")].data[1] for i in 1:dim_u]
    E_u_sample = Diagonal( E_u_vec_sample .- 1)
    
    E_v_vec_sample = [chain[sample_idx][Symbol("E_v_vec[$i]")].data[1] for i in 1:dim_v]
    E_v_sample = Diagonal( E_v_vec_sample .- 1)
    
    # --- Build U_z and V_z via the Cayley transform including the sign matrices ---
    I_dim_u = Matrix(I, dim_u, dim_u)
    U_full_sample = E_u_sample * ((I_dim_u - S_u_sample) / (I_dim_u + S_u_sample))
    U_z_sample = U_full_sample[:, 1:ℓ]
    
    I_dim_v = Matrix(I, dim_v, dim_v)
    V_full_sample = E_v_sample * ((I_dim_v - S_v_sample) / (I_dim_v + S_v_sample))
    V_z_sample = V_full_sample  # For V, we keep the full dim_v×dim_v orthogonal matrix
    
    # --- Extract singular value parameters ---
    ε = 1e-4
    sigma_vals_sample = [chain[sample_idx][Symbol("sigma_vals[$i]")].data[1] for i in 1:(ℓ-1)]
    sigma_diag = Diagonal(vcat(1 - ε, sigma_vals_sample))
    
    # --- Extract gamma ---
    γ_sample  = chain[sample_idx][:γ].data[1]
    
    # --- Construct Z and then B̃, D, G̃ ---
    Z_sample = U_z_sample * sigma_diag * transpose(V_z_sample)
    B̃_sample = γ_sample .* Z_sample[1:n, :]
    D_sample   = γ_sample .* Z_sample[n+1:n+q, :]
    G̃_sample  = γ_sample .* Z_sample[n+q+1:end, :]
    
    # --- Construct S from s_veck_sample ---
    S_sample = skew_matrix(s_veck_sample, n)
    
    # --- Extract ρ and σ ---
    ρ_sample = chain[sample_idx][:ρ].data[1]
    σ_sample = chain[sample_idx][:σ].data[1]
    
    # --- Reconstruct system matrices ---
    P_inv_sample = L_Pinv_sample * L_Pinv_sample'
    Q_mat_sample = L_Q_sample * L_Q_sample' 
    A_sample = -0.5 * P_inv_sample * (Q_mat_sample + F̃_sample' * F̃_sample + C_sample' * C_sample + S_sample)
    F_sample = L_Pinv_sample * F̃_sample
    G_sample = L_Pinv_sample * G̃_sample
    B_sample = P_inv_sample * (L_Q_sample * B̃_sample - ρ_sample * F̃_sample' * G̃_sample - C_sample' * D_sample)
    
    # --- Simulate trajectories using the extracted parameters ---
    data_sample = zeros(q, length(times_extended), num_traj_chain)
    for traj in 1:num_traj_chain
        x = zeros(n, length(times_extended))
        y = zeros(q, length(times_extended))
        x[:, 1] = zeros(n)
        for k in 1:(length(times_extended)-1)
            dt = times_extended[k+1] - times_extended[k]
            u_val = u_vals_extended[k]
            Q_noise = dt * [1.0 0.3; 0.3 1.0]
            w = rand(MvNormal(zeros(2), Q_noise))
            x[:, k+1] = x[:, k] + dt*A_sample*x[:, k] + dt*B_sample*u_val +
                        F_sample*x[:, k]*w[1] + G_sample*u_val*w[2]
        end
        for k in 1:length(times_extended)
            u_val = u_vals_extended[k]
            y[:, k] = C_sample*x[:, k] + D_sample*u_val 
        end
        data_sample[:, :, traj] = y
    end
    
    posterior_means[:, :, sample_idx] = dropdims(mean(data_sample, dims=3), dims=3)
    posterior_vars[:, :, sample_idx] = dropdims(var(data_sample, dims=3), dims=3)
end

# Compute average posterior mean and variance over samples (e.g., over samples 14000:end)
posterior_means_drop = posterior_means[:, :, 19000:end]
posterior_vars_drop = posterior_vars[:, :, 19000:end]
posterior_mean_mean = dropdims(mean(posterior_means_drop, dims=3), dims=3)
posterior_var_mean = dropdims(mean(posterior_vars_drop, dims=3), dims=3)

# SSM Direct Inference using gaussian prior
@model function ssm_direct_inference(data, inputs, times, m, n, ℓ, q)
    σ_shape = 1 
    σ_scale = 1
    # Direct priors on the matrices:
    A ~ filldist(Normal(0, 1), n, n)       # Dynamics matrix (used in μ update)
    B ~ filldist(Normal(0, 1), n, ℓ)         # Input effect on state
    F ~ filldist(Normal(0, 1), n, n)         # State noise multiplier (multiplicative noise on x)
    G ~ filldist(Normal(0, 1), n, ℓ)         # Input noise multiplier (multiplicative noise on u)
    C ~ filldist(Normal(0, 1), q, n)         # Observation matrix for state
    D ~ filldist(Normal(0, 1), q, ℓ)         # Observation matrix for input
    ρ ~ Uniform(-0.95, 0.95)                 # Correlation in the state-noise
    σ ~ InverseGamma(σ_shape, σ_scale)       # Measurement noise std
    
    A = Matrix(A)
    B = Matrix(B)
    F = Matrix(F)
    G = Matrix(G)
    C = Matrix(C)
    D = Matrix(D)

    I_n = Matrix(I(n))
    I_q = Matrix(I(q))
    dt = times[2] - times[1]
    Adis = I_n+dt*A+dt^2/2*A*A + dt^3/6*A*A*A
    Bdis = dt*B + dt^2/2*A*B + dt^3/6*A*A*B
    for traj in 1:m
        u_traj = inputs[traj]
        μ = zeros(n,1)
        Σ = zeros(n, n)
        # Loop over time steps
        for i in 1:(length(times) - 1)
            dt = times[i+1] - times[i]
            u_val = vec(u_traj(times[i]))
            μ = Adis*μ +  Bdis * u_val
            Σ = (Adis) * Σ * transpose(Adis) + dt * (F * (Σ + μ * transpose(μ)) * transpose(F) + G * (u_val * transpose(u_val)) * transpose(G) +
                 ρ * (F * μ * transpose(u_val) * transpose(G) + G * u_val * transpose(μ) * transpose(F)))
            Σ = 0.5 * (Σ + transpose(Σ))
            if any(isnan, Array(μ)) || any(isinf, Array(μ)) || any(isnan, Array(Σ)) || any(isinf, Array(Σ))
                Turing.@addlogprob!(-Inf)
            else
            y_mean = C * μ + D * u_val
            y_cov = C * Σ * transpose(C) + (σ^2+1e-4) * I_q 
            data[traj][:, i] ~ MvNormal(vec(y_mean), y_cov)
            end
        end
    end
end
model_direct = ssm_direct_inference(data, inputs, times, m, n, ℓ, q)
MAP_estimate_direct = optimize(model_direct, MAP(),  LBFGS(),Optim.Options(iterations=1_000, allow_f_increases=true))
chain_direct = sample(model_direct, HMC(0.01, 3; adtype=AutoReverseDiff(true)), init_params = MAP_estimate_direct.values.array, 20000) 

save("chain_direct_20000_HMC_4statesQ_newM10.jld2", "chain_direct", chain_direct)
#chain_direct = load("chain_direct_20000_HMC_4statesQ_10.jld2", "chain_direct")
# ----------------------------
# Posterior simulation for the direct model:
# Extend the observation times
# ----------------------------
times_extended = collect(0.0:0.0025:2)
num_traj_true = 100  # number of new trajectories for computing true statistics
u_vals_extended = [u_func(t) for t in times_extended]

# Preallocate arrays to store the true trajectories (as before)
data_true = zeros(q, length(times_extended), num_traj_true)
for traj in 1:num_traj_true
    x = zeros(n, length(times_extended))
    y = zeros(q, length(times_extended))
    x[:, 1] = zeros(n)
    for k in 1:(length(times_extended)-1)
        dt = times_extended[k+1] - times_extended[k]
        u_val = u_vals_extended[k]
        Q_noise = dt * [1.0 ρ; ρ 1.0]
        w = rand(MvNormal(zeros(2), Q_noise))
        x[:, k+1] = x[:, k] + dt*A*x[:, k] + dt*B*u_val + F*x[:, k]*w[1] + G*u_val*w[2]
    end
    for k in 1:length(times_extended)
        u_val = u_vals_extended[k]
        y[:, k] = C*x[:, k] + D*u_val + rand(MvNormal(zeros(q), (σ^2)*I(q)))
    end
    data_true[:, :, traj] = y
end
true_mean = dropdims(mean(data_true, dims=3), dims=3)
true_var = dropdims(var(data_true, dims=3), dims=3)

# ----------------------------
# Compute posterior mean and variance trajectories from the direct model’s chain.
# (Assuming chain_direct is available from the inference run above)
# ----------------------------
num_samples = length(chain_direct)   # number of MCMC samples
num_traj_chain = 100   # simulated trajectories per chain sample
posterior_means_direct = zeros(q, length(times_extended), num_samples)
posterior_vars_direct = zeros(q, length(times_extended), num_samples)

for sample_idx in 19000:num_samples
    # Extract parameters directly from the chain sample.
    A_sample = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A_sample[i,j] = chain_direct[sample_idx][Symbol("A[$i, $j]")].data[1]
        end
    end
    
    B_sample = zeros(n, ℓ)
    for i in 1:n
        for j in 1:ℓ
            B_sample[i,j] = chain_direct[sample_idx][Symbol("B[$i, $j]")].data[1]
        end
    end

    F_sample = zeros(n, n)
    for i in 1:n
        for j in 1:n
            F_sample[i,j] = chain_direct[sample_idx][Symbol("F[$i, $j]")].data[1]
        end
    end

    G_sample = zeros(n, ℓ)
    for i in 1:n
        for j in 1:ℓ
            G_sample[i,j] = chain_direct[sample_idx][Symbol("G[$i, $j]")].data[1]
        end
    end

    C_sample = zeros(q, n)
    for i in 1:q
        for j in 1:n
            C_sample[i,j] = chain_direct[sample_idx][Symbol("C[$i, $j]")].data[1]
        end
    end

    D_sample = zeros(q, ℓ)
    for i in 1:q
        for j in 1:ℓ
            D_sample[i,j] = chain_direct[sample_idx][Symbol("D[$i, $j]")].data[1]
        end
    end
    ρ_sample = chain_direct[sample_idx][:ρ][1]
    σ_sample = chain_direct[sample_idx][:σ][1]
    
    # Simulate trajectories for this posterior sample
    data_sample = zeros(q, length(times_extended), num_traj_chain)
    for traj in 1:num_traj_chain
        x = zeros(n, length(times_extended))
        y = zeros(q, length(times_extended))
        x[:, 1] = zeros(n)
        for k in 1:(length(times_extended)-1)
            dt = times_extended[k+1] - times_extended[k]
            u_val = u_vals_extended[k]
            Q_noise = dt * [1.0 ρ_sample; ρ_sample 1.0]
            w = rand(MvNormal(zeros(2), Q_noise))
            x[:, k+1] = x[:, k] + dt*(A_sample*x[:, k] + B_sample*u_val) +
                        F_sample*x[:, k]*w[1] + G_sample*u_val*w[2]
        end
        for k in 1:length(times_extended)
            u_val = u_vals_extended[k]
            y[:, k] = C_sample*x[:, k] + D_sample*u_val +
                      rand(MvNormal(zeros(q), (σ_sample^2)*I(q)))
        end
        data_sample[:, :, traj] = y
    end
    
    posterior_means_direct[:, :, sample_idx] = dropdims(mean(data_sample, dims=3), dims=3)
    posterior_vars_direct[:, :, sample_idx] = dropdims(var(data_sample, dims=3), dims=3)
end

# Average over posterior samples (e.g., samples 1501:end)
posterior_means_direct_drop = posterior_means_direct[:, :, 19000:end]
posterior_vars_direct_drop = posterior_vars_direct[:, :, 19000:end]
posterior_mean_direct_mean = dropdims(mean(posterior_means_direct_drop, dims=3), dims=3)
posterior_var_direct_mean = dropdims(mean(posterior_vars_direct_drop, dims=3), dims=3)


# ----------------------------
# Posterior predictive simulation from the chain
# ----------------------------
num_samples = size(chain, 1)
num_traj_chain = 100  # Trajectories to simulate per chain sample

posterior_means = zeros(q, length(times_extended), num_samples)
posterior_vars = zeros(q, length(times_extended), num_samples)

# Loop over (a subset of) posterior samples
for sample_idx in 19000:num_samples
    # --- Extract L_Pinv parameters ---
    diag_val_p = [chain[sample_idx][Symbol("diag_val_p[$i]")].data[1] for i in 1:n]
    L_Pinv_sample = zeros(n, n)
    for i in 1:n
        L_Pinv_sample[i, i] = sqrt(diag_val_p[i])
        for j in 1:(i-1)
            L_Pinv_sample[i, j] = chain[sample_idx][Symbol("L_Pinv[$i, $j]")].data[1]
        end
    end
    L_Pinv_sample = L_Sigma_p * L_Pinv_sample
    
    # --- Extract L_Q parameters ---
    diag_val_q = [chain[sample_idx][Symbol("diag_val_q[$i]")].data[1] for i in 1:n]
    L_Q_sample = zeros(n, n)
    for i in 1:n
        L_Q_sample[i, i] = sqrt(diag_val_q[i])
        for j in 1:(i-1)
            L_Q_sample[i, j] = chain[sample_idx][Symbol("L_Q[$i, $j]")].data[1]
        end
    end
    L_Q_sample = L_Sigma_q * L_Q_sample

    # --- Extract S (via its vector s_veck) ---
    s_veck_sample = [chain[sample_idx][Symbol("s_veck[$i]")].data[1] for i in 1:div(n*(n-1),2)]
    
    # --- Extract F̃ and C ---
    vec_F̃_sample = [chain[sample_idx][Symbol("vec_F̃[$i]")].data[1] for i in 1:(n*n)]
    F̃_sample = reshape(vec_F̃_sample, n, n)
    
    vec_C_sample = [chain[sample_idx][Symbol("vec_C[$i]")].data[1] for i in 1:(q*n)]
    C_sample = reshape(vec_C_sample, q, n)
    
    # --- Extract the Cayley transform parameters for [B̃; D; G̃] ---
    # Set dimensions: dim_u = n+q+n (here 5) and dim_v = ℓ (here 2)
    dim_u = n + q + n
    num_u = div(dim_u*(dim_u-1), 2)
    u_veck_sample = [chain[sample_idx][Symbol("s_u[$i]")].data[1] for i in 1:num_u]
    S_u_sample = skew_matrix(u_veck_sample, dim_u)
    
    dim_v = ℓ
    num_v = div(dim_v*(dim_v-1), 2)
    if num_v > 0
        v_veck_sample = [chain[sample_idx][Symbol("s_v[$i]")].data[1] for i in 1:num_v]
        S_v_sample = skew_matrix(v_veck_sample, dim_v)
    else
        S_v_sample = Matrix(I, dim_v, dim_v)
    end
    
    # --- Extract sign vectors and build sign matrices ---
    E_u_vec_sample = [chain[sample_idx][Symbol("E_u_vec[$i]")].data[1] for i in 1:dim_u]
    E_u_sample = Diagonal( E_u_vec_sample .- 1)
    
    E_v_vec_sample = [chain[sample_idx][Symbol("E_v_vec[$i]")].data[1] for i in 1:dim_v]
    E_v_sample = Diagonal( E_v_vec_sample .- 1)
    
    # --- Build U_z and V_z via the Cayley transform including the sign matrices ---
    I_dim_u = Matrix(I, dim_u, dim_u)
    U_full_sample = E_u_sample * ((I_dim_u - S_u_sample) / (I_dim_u + S_u_sample))
    U_z_sample = U_full_sample[:, 1:ℓ]
    
    I_dim_v = Matrix(I, dim_v, dim_v)
    V_full_sample = E_v_sample * ((I_dim_v - S_v_sample) / (I_dim_v + S_v_sample))
    V_z_sample = V_full_sample  # For V, we keep the full dim_v×dim_v orthogonal matrix
    
    # --- Extract singular value parameters ---
    ε = 1e-4
    sigma_vals_sample = [chain[sample_idx][Symbol("sigma_vals[$i]")].data[1] for i in 1:(ℓ-1)]
    sigma_diag = Diagonal(vcat(1 - ε, sigma_vals_sample))
    
    # --- Extract gamma ---
    γ_sample  = chain[sample_idx][:γ].data[1]
    
    # --- Construct Z and then B̃, D, G̃ ---
    Z_sample = U_z_sample * sigma_diag * transpose(V_z_sample)
    B̃_sample = γ_sample .* Z_sample[1:n, :]
    D_sample   = γ_sample .* Z_sample[n+1:n+q, :]
    G̃_sample  = γ_sample .* Z_sample[n+q+1:end, :]
    
    # --- Construct S from s_veck_sample ---
    S_sample = skew_matrix(s_veck_sample, n)
    
    # --- Extract ρ and σ ---
    ρ_sample = chain[sample_idx][:ρ].data[1]
    σ_sample = chain[sample_idx][:σ].data[1]
    
    # --- Reconstruct system matrices ---
    P_inv_sample = L_Pinv_sample * L_Pinv_sample'
    Q_mat_sample = L_Q_sample * L_Q_sample' 
    A_sample = -0.5 * P_inv_sample * (Q_mat_sample + F̃_sample' * F̃_sample + C_sample' * C_sample + S_sample)
    F_sample = L_Pinv_sample * F̃_sample
    G_sample = L_Pinv_sample * G̃_sample
    B_sample = P_inv_sample * (L_Q_sample * B̃_sample - ρ_sample * F̃_sample' * G̃_sample - C_sample' * D_sample)
    
    # --- Simulate trajectories using the extracted parameters ---
    data_sample = zeros(q, length(times_extended), num_traj_chain)
    for traj in 1:num_traj_chain
        x = zeros(n, length(times_extended))
        y = zeros(q, length(times_extended))
        x[:, 1] = zeros(n)
        for k in 1:(length(times_extended)-1)
            dt = times_extended[k+1] - times_extended[k]
            u_val = u_vals_extended[k]
            Q_noise = dt * [1.0 0.3; 0.3 1.0]
            w = rand(MvNormal(zeros(2), Q_noise))
            x[:, k+1] = x[:, k] + dt*A_sample*x[:, k] + dt*B_sample*u_val +
                        F_sample*x[:, k]*w[1] + G_sample*u_val*w[2]
        end
        for k in 1:length(times_extended)
            u_val = u_vals_extended[k]
            y[:, k] = C_sample*x[:, k] + D_sample*u_val 
        end
        data_sample[:, :, traj] = y
    end
    
    posterior_means[:, :, sample_idx] = dropdims(mean(data_sample, dims=3), dims=3)
    posterior_vars[:, :, sample_idx] = dropdims(var(data_sample, dims=3), dims=3)
end

# Compute average posterior mean and variance over samples (e.g., over samples 14000:end)
posterior_means_drop = posterior_means[:, :, 19000:end]
posterior_vars_drop = posterior_vars[:, :, 19000:end]
posterior_mean_mean = dropdims(mean(posterior_means_drop, dims=3), dims=3)
posterior_var_mean = dropdims(mean(posterior_vars_drop, dims=3), dims=3)

# ----------------------------
# Plot comparisons for the direct model:
# Posterior mean trajectory vs. true mean trajectory
# ----------------------------
pmean_direct = plot(times_extended, posterior_means_direct_drop[1,:,:],
    alpha=0.01, label="", xlabel="Time", ylabel="Observation Mean",
    title="Direct Model: Mean Trajectory Comparison", xlims=(0, 1), ylims=(-10, 10), color=:blue)
pmean_direct = plot!(pmean_direct,times_extended, posterior_mean_direct_mean[1,:], label="Posterior Mean", lw=2, color=:red)
pmean_direct = vline!(pmean_direct,[0.2], label="Training End", linestyle=:dash)
pmean_direct = plot!(pmean_direct, times_extended, true_mean[1,:], label="True Mean", lw=2, linestyle=:dash, color=:black)

# Posterior variance trajectory vs. true variance trajectory
pvar_direct = plot(times_extended, posterior_vars_direct_drop[1,:,:],
    alpha=0.01, label="", xlabel="Time", ylabel="Observation Variance",
    title="Direct Model: Variance Trajectory Comparison", color=:blue, xlims=(0, 1), ylims=(0, 8))
plot!(times_extended, posterior_var_direct_mean[1,:], label="Posterior Mean Variance", lw=2, color=:red)
vline!([0.2], label="Training End", linestyle=:dash)
plot!(times_extended, true_var[1,:], label="True Variance", lw=2, linestyle=:dash, color=:black)


#save("results4states10M.jld2", "times_extended", times_extended, "posterior_means_drop", posterior_means_drop, "posterior_vars_drop", posterior_vars_drop, "posterior_mean_mean", posterior_mean_mean, "posterior_var_mean", posterior_var_mean,  "posterior_means_direct_drop", posterior_means_direct_drop,  "posterior_vars_direct_drop", posterior_vars_direct_drop, "posterior_mean_direct_mean", posterior_mean_direct_mean, "posterior_var_direct_mean", posterior_var_direct_mean,  "true_mean", true_mean, "true_var", true_var)
datares = load("results4states10M.jld2")
times_extended = datares["times_extended"]
posterior_means_drop = datares["posterior_means_drop"]
posterior_vars_drop = datares["posterior_vars_drop"]
posterior_mean_mean = datares["posterior_mean_mean"]
posterior_var_mean = datares["posterior_var_mean"]
posterior_means_direct_drop = datares["posterior_means_direct_drop"]
posterior_vars_direct_drop = datares["posterior_vars_direct_drop"]
posterior_mean_direct_mean = datares["posterior_mean_direct_mean"]
posterior_var_direct_mean = datares["posterior_var_direct_mean"]
true_mean = datares["true_mean"]
true_var = datares["true_var"]

pgfplotsx()
#gr()
#Plots for the paper: 
using LaTeXStrings
Sp = 5;
SS = 1;
pmean1 = plot(times_extended[1:Sp:end], posterior_means_drop[1,1:Sp:end,500:end], alpha=0.01, color=:blue, label="",  xlims=(0, 2), ylims=(-8, 10),  tex_output_standalone = true, title = L"$Q$-WNS-BRL-Cayley prior", titlefontsize= 10,size = (450,380))
pmean1 = plot!(pmean1,times_extended[1:Sp:end], posterior_mean_mean[1,1:Sp:end], label="Post. Mean", linewidth=1, color=:red)
pmean1 = vline!(pmean1, [0.5], label="Inference end", linestyle=:dash, color=:green,lw=2)
pmean1 = plot!(pmean1, times_extended[1:Sp:end], true_mean[1,1:Sp:end], label="True Mean", linewidth=1, color=:black,linestyle=:dash)
pmean1 = xlabel!(pmean1, "Time")
pmean1 = ylabel!(pmean1, "Output (mean)")
#pmean = title!(pmean, "Stable parametrization")
pmean1 = plot!(pmean1, legend=:top, legendcolumns=3, foreground_color_legend = nothing, background_color_legend = nothing, titlefontsize= 10)

pmean1_direct = plot(times_extended[1:Sp:end], posterior_means_direct_drop[1,1:Sp:end,500:end],
    alpha=0.01, label="", xlabel="Time", ylabel="Output (mean)", title="Free gaussian prior", xlims=(0, 2), ylims=(-8, 10), color=:blue, size = (450,380), titlefontsize= 10)
pmean1_direct = plot!(pmean1_direct,times_extended[1:Sp:end], posterior_mean_direct_mean[1,1:Sp:end], label="", lw=1, color=:red)
pmean1_direct = vline!(pmean1_direct,[0.5], label="", linestyle=:dash, color=:green, lw=2)
pmean1_direct = plot!(pmean1_direct, times_extended[1:Sp:end], true_mean[1,1:Sp:end], label="", lw=1, linestyle=:dash, color=:black)

pmean1_combined = plot(pmean1,pmean1_direct,layout=(2,1), size = (450,380), xguidefontsize=10, yguidefontsize = 10)



#Code for to generate the plots for the poster: 
using Statistics, Measures
# ================= styling =================
default(
    guidefontcolor = AAU_BLUE,
    tickfontcolor  = AAU_BLUE,
    foreground_color_subplot = AAU_BLUE,
    background_color = nothing,
    grid = :y, gridalpha = 0.15, gridcolor = AAU_BLUE,
    tickfontsize   = 28,
    guidefontsize  = 30,   # axis labels
    legendfontsize = 26,
    titlefontsize  = 32,
)

# ================= helpers =================
# Row-wise sample quantiles for a T×S matrix (rows=time, cols=samples)
function row_quantiles(Y::AbstractMatrix, probs::AbstractVector{<:Real})
    T = size(Y,1)
    Q = Matrix{Float64}(undef, length(probs), T)
    @inbounds for i in 1:T
        @views r = Y[i, :]
        for (j,p) in enumerate(probs)
            Q[j,i] = quantile(r, p)
        end
    end
    Q
end

# One subplot with 99/95/75% ribbons around the median + optional overlays
function quantile_ribbons_panel(
    t, Y;
    title_text::AbstractString,
    xlims::Tuple, ylims::Tuple,
    ylabel_text::AbstractString,
    xlabel_text::AbstractString,
    color_band = AAU_G,
    post_line = nothing,       # vector length T
    true_line = nothing,       # vector length T
    inference_time = 0.5,
    size=(540,420)
)
    # central bands (equal-tailed)
    bands = [
        (0.005, 0.995, 0.10),  # 99%
        (0.025, 0.975, 0.16),  # 95%
        (0.125, 0.875, 0.24),  # 75%
    ]
    needed = sort(unique(vcat(0.5, [p for (a,b,_) in bands for p in (a,b)])))
    Q = row_quantiles(Y, needed)
    idx = Dict(p => i for (i,p) in enumerate(needed))
    median_line = Q[idx[0.5], :]

    p = plot(t, median_line;
        label="", lw=3, color=color_band,
        xlims=xlims, ylims=ylims, size=size,
        title=title_text, titlefontsize=32, titlefontcolor=AAU_BLUE,
        xlabel=xlabel_text, ylabel=ylabel_text,
        legend=false
    )

    # ribbons (draw widest → narrowest)
    for (lo, hi, fa) in sort(bands; by = x -> x[2]-x[1], rev=true)
        qL = Q[idx[lo], :]; qU = Q[idx[hi], :]
        plot!(p, t, median_line,
              ribbon=(median_line .- qL, qU .- median_line),
              color=color_band, fillalpha=fa, label="")
    end

    # overlays
    if post_line !== nothing
        plot!(p, t, post_line; label="", lw=3, color=AAU_GREEN)
    end
    if inference_time !== nothing
        vline!(p, [inference_time]; label="", linestyle=:dash, color=AAU_GOLD, lw=4)
    end
    if true_line !== nothing
        plot!(p, t, true_line; label="", lw=2.0, linestyle=:dash, color=AAU_BLACK)
    end

    return p
end

# ================= data slices (use your arrays) =================
Sp = 1
t  = times_extended[1:Sp:end]

# Means (T×S)
Y_mean_stable = posterior_means_drop[1, 1:Sp:end, 500:end]
Y_mean_free   = posterior_means_direct_drop[1, 1:Sp:end, 500:end]
post_mean_stable = posterior_mean_mean[1, 1:Sp:end]
post_mean_free   = posterior_mean_direct_mean[1, 1:Sp:end]
true_mean_line   = true_mean[1, 1:Sp:end]

# Variances (T×S) — clip “direct” like before
Y_var_stable = posterior_vars_drop[1, 1:Sp:end, 500:end]
Y_var_free   = copy(posterior_vars_direct_drop[1, 1:Sp:end, 500:end]);  Y_var_free[Y_var_free .> 16] .= 16
post_var_stable = posterior_var_mean[1, 1:Sp:end]
post_var_free   = copy(posterior_var_direct_mean[1, 1:Sp:end]);         post_var_free[post_var_free .> 16] .= 16
true_var_line   = true_var[1, 1:Sp:end]

# Axis limits
xlimv = 1.2

xlims_mean = (0.0, xlimv); ylims_mean = (-8.0, 10.0)
xlims_var  = (0.0, xlimv);  ylims_var  = (0.0, 15.0)
i_end = searchsortedlast(t, xlimv)
keep  = 1:i_end

t    = t[keep]
Y_mean_stable    = Y_mean_stable[keep, :]
Y_mean_free      = Y_mean_free[keep, :]
post_mean_stable = post_mean_stable[keep]
post_mean_free   = post_mean_free[keep]
true_mean_line   = true_mean_line[keep]

Y_var_stable     = Y_var_stable[keep, :]
Y_var_free       = Y_var_free[keep, :]
post_var_stable  = post_var_stable[keep]
post_var_free    = post_var_free[keep]
true_var_line    = true_var_line[keep]
# ================= four panels (top row titles include column headings) =================
p11 = quantile_ribbons_panel(
    t, Y_mean_stable;
    title_text = "Mean of 100 simulated realizations (stable SSM prior)",
    xlims=xlims_mean, ylims=ylims_mean, ylabel_text="Output (mean)",xlabel_text="",
    color_band=AAU_G,
    post_line=post_mean_stable, true_line=true_mean_line
)

p21 = quantile_ribbons_panel(
    t, Y_mean_free;
    title_text = "Free Gaussian prior",
    xlims=xlims_mean, ylims=ylims_mean, ylabel_text="Output (mean)",xlabel_text="Time",
    color_band=AAU_G,
    post_line=post_mean_free, true_line=true_mean_line
)

p12 = quantile_ribbons_panel(
    t, Y_var_stable;
    title_text = "Variance of 100 simulated realizations (stable SSM prior)",
    xlims=xlims_var, ylims=ylims_var, ylabel_text="Output (var.)",xlabel_text="",
    color_band=AAU_G,
    post_line=post_var_stable, true_line=true_var_line
)

p22 = quantile_ribbons_panel(
    t, Y_var_free;
    title_text = "Free Gaussian prior",
    xlims=xlims_var, ylims=ylims_var, ylabel_text="Output (var.)",xlabel_text="Time",
    color_band=AAU_G,
    post_line=post_var_free, true_line=true_var_line
)

# ================= legend-only subplot (centered, full width, no geometry drawn) =================
legend_only = plot(
    ; legend=:bottom, framestyle=:none, grid=false,
      xaxis=false, yaxis=false, xticks=false, yticks=false,
      xlim=(0,1), ylim=(0,1)
)

# Put all legend entries outside the visible y-limits so nothing is drawn,
# but the legend swatches are created.
xleg = [0.0, 1.0]; yoff = 2.0; yleg = fill(yoff, 2)

# Lines
plot!(legend_only, xleg, yleg; color=AAU_BLUE,  lw=3,            label="Median")
plot!(legend_only, xleg, yleg; color=AAU_GREEN, lw=3,            label="Posterior mean")
plot!(legend_only, xleg, yleg; color=AAU_GOLD,  lw=4, linestyle=:dash, label="Inference end")
plot!(legend_only, xleg, yleg; color=AAU_BLACK, lw=2, linestyle=:dash, label="True")

# Ribbons (just to show swatches in the legend)
plot!(legend_only, xleg, yleg; color=AAU_BLUE, ribbon=(fill(0.90,2), fill(0.90,2)), fillalpha=0.10, label="99% band")
plot!(legend_only, xleg, yleg; color=AAU_BLUE, ribbon=(fill(0.50,2), fill(0.50,2)), fillalpha=0.16, label="95% band")
plot!(legend_only, xleg, yleg; color=AAU_BLUE, ribbon=(fill(0.25,2), fill(0.25,2)), fillalpha=0.24, label="75% band")

plot!(legend_only; legendcolumns=7,
      foreground_color_legend=nothing, background_color_legend=nothing)

# ================= final 2×2 grid + full-width centered legend row =================
# No margins here (avoids the AbsoluteLength + Int error in PGFPlotsX).
lay = @layout [a{0.13h};grid(2, 2; widths=[0.5, 0.5], heights=[0.5, 0.5])]

final_combined = plot(
    legend_only, p11, p12, p21, p22;
    layout = lay,
    size = (2800, 900),
    xguidefontsize=30, yguidefontsize=30,
    tex_output_standalone = false, bottom_margin = 10mm, left_margin = 30mm
)
#savefig(final_combined, "both_exampleS.tikz")