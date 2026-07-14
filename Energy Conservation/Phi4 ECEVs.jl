#Code for calculating the expectation values of the energy density and energy flux at x=0 in order to check energy conservation

using ITensors, ITensorMPS, HDF5

### Initial Parameters ###
N = parse(Int, ARGS[1]) #Number of lattice sites per dimension
d = 1 #Number of spatial dimensions
Dim = parse(Int, ARGS[2]) #Truncated local Hilbert space dimension
a = parse(Float64, ARGS[3]) #Lattice spacing

n_0 = round(Int, ((N-1)/2)) #Index of the point at the center of the lattice (0-based indexing).

mass = parse(Float64, ARGS[4])
m0 = parse(Float64, ARGS[5]) #Basis frequency
l = parse(Float64, ARGS[6]); #phi^4 coupling strength

### Field Operator $\phi(\mathbf{x})$ and $\pi(\mathbf{x})$ ###
function a_matrix(D)
    A = zeros(D, D)
    for n in 1:D-1
        A[n, n+1] = sqrt(n)
    end
    return A
end

phi_matrix = D -> (a_matrix(D) + a_matrix(D)')/sqrt(2*m0)
pi_matrix = D -> im*sqrt(m0/2)*(a_matrix(D)' - a_matrix(D))

ITensors.op(::OpName"phi", ::SiteType"Boson", D::Int) = phi_matrix(D)

ITensors.op(::OpName"phi2", ::SiteType"Boson", D::Int) = phi_matrix(D)^2

ITensors.op(::OpName"phi4", ::SiteType"Boson", D::Int) = phi_matrix(D)^4

ITensors.op(::OpName"pi", ::SiteType"Boson", D::Int) = pi_matrix(D)

ITensors.op(::OpName"pi2", ::SiteType"Boson", D::Int) = pi_matrix(D)^2

### Hamiltonian ###
H0 = OpSum() #Non-interacting Hamiltonian OpSum
for x in 1:N
    global H0 += a^d/2, "pi2", x
    global H0 += a^d * d/a^2, "phi2", x
    if x < N
        global H0 -= a^d * 1/a^2, "phi", x, "phi", x+1
    else
        global H0 -= a^d * 1/a^2, "phi", x, "phi", 1
    end
    global H0 += a^d/2 * mass^2, "phi2", x
end

HInt = OpSum() #Interacting Hamiltonian OpSum
for x in 1:N
    global HInt += l/factorial(4) * a^d, "phi4", x
end

H_OS = H0 + HInt #Full Hamiltonian OpSum

if parse(Bool, ARGS[10])
    EEC_MPS = h5open("1d_EEC_MPS/N=$N,a=$a,dim=$Dim,l=$l,m=$mass", "r")
    sites = read(EEC_MPS, "sites", Vector{Index{Int64}})
    H = MPO(H_OS, sites);
    vac = read(EEC_MPS, "vac", MPS)
    close(EEC_MPS)
else
    sites = siteinds("Boson", N; dim=Dim); #Create ITensor sites

    H = MPO(H_OS, sites); #Hamiltonian MPO

    ### Vacuum State MPS ###
    psi0 = random_mps(sites;linkdims=10)
    nsweeps = 6
    maxdim = [10,20,100,100,200,200]
    cutoff = [1E-10]
    energy, vac = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    EEC_MPS = h5open("1d_EEC_MPS/N=$N,a=$a,dim=$Dim,l=$l,m=$mass", "w")
    write(EEC_MPS, "sites", sites)
    write(EEC_MPS, "vac", vac)
    close(EEC_MPS)
end

### Energy Flux Operator ###
#Function for applying the T_{0i}(x,t) operator.
#Params: state to apply T_{0i} on, time of measurement, position of detector (as a tuple)
function T0i(state, t, r; dir=0, nsteps=1, maxdim=200, cutoff=1e-8)
    psi = copy(state)

    #Apply exp(-iHt). (I am using the forward time evolution convention, but 2604.26226 uses the inverse time evolution convention.)
    psi = tdvp(H, -im*t, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, normalize=false)

    #Apply T_{0i}(x). (Specifically, T_{01}(x) in 1+1d)
    x = r[1]

    T0x_OS = OpSum()
    if x > 0 || (x == 0 && dir > 0)
        T0x_OS += 1/2 * a^(d-2), "phi", (n_0+1) + x, "pi", (n_0+1) + x+1
        T0x_OS -= 1/2 * a^(d-2), "phi", (n_0+1) + x+1, "pi", (n_0+1) + x
    elseif x < 0 || (x == 0 && dir < 0)
        T0x_OS += 1/2 * a^(d-2), "phi", (n_0+1) + x, "pi", (n_0+1) + x-1
        T0x_OS -= 1/2 * a^(d-2), "phi", (n_0+1) + x-1, "pi", (n_0+1) + x
    end

    T0i_MPO = MPO(T0x_OS, sites)
    psi = apply(T0i_MPO, psi)

    #Apply exp(iHt)
    psi = tdvp(H, im*t, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, normalize=false)

    return psi
end

### Check Energy Conservation ###
t_i = parse(Float64, ARGS[7])
t_max = parse(Float64, ARGS[8])
dt = parse(Float64, ARGS[9])
t_range = range(t_i, t_max, step=dt)

#Define the excited state, \phi(0)\ket{\omega}
phi0_OS = OpSum()
phi0_OS += "phi", n_0+1
phi0 = MPO(phi0_OS, sites)

psi = apply(phi0, vac) #Excited state
norm_psi = inner(psi, psi)

H_0_OS = OpSum() #Hamiltonian OpSum at index n_0 (i.e. x=0)
H_0_OS += a^d/2, "pi2", n_0+1
H_0_OS += a^d/2 * mass^2, "phi2", n_0+1
H_0_OS += l/factorial(4) * a^d, "phi4", n_0+1
H_0_OS += a^d * 1/(a^2), "phi2", n_0+1
H_0_OS -= 1/2 * a^d * 1/a^2, "phi", n_0+1, "phi", (n_0+1)+1
H_0_OS -= 1/2 * a^d * 1/a^2, "phi", n_0+1, "phi", (n_0+1)-1
H_n_0 = MPO(H_0_OS , sites)

function CalcEdens0(t; nsteps=1, maxdim=200, cutoff=1e-8)
    psi_t = tdvp(H, -im*t, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, normalize=false)
    evH = inner(psi_t', H_n_0, psi_t) / norm_psi
    return evH
end

Edens0 = ComplexF64[]
EfluxR = ComplexF64[]
EfluxL = ComplexF64[]
for t in t_range
    evH = CalcEdens0(t)
    evR = inner(psi, T0i(psi, t, 0, dir=1)) / norm_psi
    evL = inner(psi, T0i(psi, t, 0, dir=-1)) / norm_psi
    push!(Edens0, evH)
    push!(EfluxR, evR)
    push!(EfluxL, evL)
end

EEC_data = h5open("1d_EEC_data/EVs,N=$N,a=$a,l=$l,m=$mass,ti=$t_i,tf=$t_max", "w")
EEC_data["Edens0"] = Edens0
EEC_data["EfluxR"] = EfluxR
EEC_data["EfluxL"] = EfluxL
close(EEC_data)