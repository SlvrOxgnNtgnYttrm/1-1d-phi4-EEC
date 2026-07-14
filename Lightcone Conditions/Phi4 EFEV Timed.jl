#Code for calculating the expectation value of energy flux at variable x in order to check lightcone conditions

using ITensors, ITensorMPS, LinearAlgebra, HDF5
ITensors.disable_threaded_blocksparse() #Disables block sparse multithreading
BLAS.set_num_threads(parse(Int, ARGS[1])) #Enables multithreading with BLAS

### Initial Parameters ###
N = parse(Int, ARGS[2]) #Number of lattice sites per dimension
d = 1 #Number of spatial dimensions
Dim = parse(Int, ARGS[3]) #Truncated local Hilbert space dimension
a = parse(Float64, ARGS[4]) #Lattice spacing

n_0 = round(Int, ((N-1)/2)) #Index of the point at the center of the lattice (0-based indexing).

mass = parse(Float64, ARGS[5])
m0 = parse(Float64, ARGS[6]) #Basis frequency
l = parse(Float64, ARGS[7]); #phi^4 coupling strength

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

if parse(Bool, ARGS[12])
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
    psi, time, etc... = @timed tdvp(H, -im*t, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, normalize=false)
    open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "a") do io
        println(io, "t=$t - Apply exp(-iHt): $time s")
        println(io, ". . . - B. dims: $(linkdims(psi))")
    end

    #Apply T_{0i}(x). (Specifically, T_{01}(x) in 1+1d)
    psi, time, etc... = @timed begin
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
        apply(T0i_MPO, psi)
    end
    open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "a") do io
        println(io, "t=$t - Apply T_{0i}(x): $time s")
        println(io, ". . . - B. dims: $(linkdims(psi))")
    end

    #Apply exp(iHt)
    psi, time, etc... = @timed tdvp(H, im*t, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, normalize=false)
    open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "a") do io
        println(io, "t=$t - Apply exp(iHt): $time s")
        println(io, ". . . - B. dims: $(linkdims(psi))")
    end

    return psi
end

### Calculate expectation value of the energy flux operator ###
r = parse(Int, ARGS[8])
t_i = parse(Float64, ARGS[9])
t_max = parse(Float64, ARGS[10])
dt = parse(Float64, ARGS[11])
t_range = range(t_i, t_max, step=dt)

#Define the excited state, \phi(0)\ket{\omega}
phi0_OS = OpSum()
phi0_OS += "phi", n_0+1
phi0 = MPO(phi0_OS, sites)

psi = apply(phi0, vac) #Excited state
norm_psi = inner(psi, psi)

Eflux = ComplexF64[]
open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "w") do io
        println(io, "Elapsed Time - N=$N, a=$a, l=$l, m=$mass, r=$r, ti=$t_i, tf=$t_max, dt=$dt")
    end
for t in t_range
    if r >= 0
        ev, time, etc... = @timed inner(psi, T0i(psi, t, r, dir=1)) / norm_psi
        open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "a") do io
            println(io, "t=$t - Calc. exp. val.: $time s")
        end
    else
        ev, time, etc... = @timed inner(psi, T0i(psi, t, r, dir=-1)) / norm_psi
        open("1d_EEC_data/EFEV_Timed,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt.txt", "a") do io
            println(io, "t=$t - Calc. exp. val.: $time s")
        end
    end
    push!(Eflux, ev)
end

EEC_data = h5open("1d_EEC_data/EFEV,N=$N,a=$a,l=$l,m=$mass,r=$r,ti=$t_i,tf=$t_max,dt=$dt", "w")
EEC_data["Eflux"] = Eflux
close(EEC_data)