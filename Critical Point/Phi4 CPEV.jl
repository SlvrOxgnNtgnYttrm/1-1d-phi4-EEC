#Code for calculating the vacuum exp. val. of \phi with respect to \lambda/m_R^2

using ITensors, ITensorMPS, LinearAlgebra, SpecialFunctions, HDF5
ITensors.disable_threaded_blocksparse() #Disables block sparse multithreading
BLAS.set_num_threads(parse(Int, ARGS[1])) #Enables multithreading with BLAS

### Initial Parameters ###
N = parse(Int, ARGS[2]) #Number of lattice sites per dimension
d = 1 #Number of spatial dimensions
Dim = parse(Int, ARGS[3]) #Truncated local Hilbert space dimension
a = parse(Float64, ARGS[4]) #Lattice spacing

n_0 = round(Int, ((N-1)/2)) #Index of the point at the center of the lattice (0-based indexing).

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

sites = siteinds("Boson", N; dim=Dim); #Create ITensor sites

m_i = parse(Float64, ARGS[7]) #first renormalized mass
m_f = parse(Float64, ARGS[8]) #last renormalized mass
len = parse(Int, ARGS[9])
m_range = 10 .^ range(log10(m_i), log10(m_f), length=len);

VEVs = Float64[]
for m_R in m_range
    corr = l/2 * 1/pi * 1/sqrt(a^2*m_R^2 + 4) * ellipk(2/sqrt(a^2*m_R^2 + 4))
    mass2 = m_R^2 - corr

    ### Hamiltonian ###
    H0 = OpSum() #Non-interacting Hamiltonian OpSum
    for x in 1:N
        H0 += a^d/2, "pi2", x
        H0 += a^d * d/a^2, "phi2", x
        if x < N
            H0 -= a^d * 1/a^2, "phi", x, "phi", x+1
        else
            H0 -= a^d * 1/a^2, "phi", x, "phi", 1
        end
        H0 += a^d/2 * mass2, "phi2", x
    end

    HInt = OpSum() #Interacting Hamiltonian OpSum
    for x in 1:N
        HInt += l/factorial(4) * a^d, "phi4", x
    end

    H_OS = H0 + HInt #Full Hamiltonian OpSum

    H = MPO(H_OS, sites); #Hamiltonian MPO

    ### Vacuum State MPS ###
    psi0 = random_mps(sites;linkdims=50)
    nsweeps = 150
    maxdim = 200#= [200,200,200,200,200,
    300,300,300,300,300,
    500,500,500,500,500,
    800,800,800,800,800,
    800,800,800,800,800,
    1000,1000,1000,1000,1000] =#
    cutoff = 1e-8
    energy, vac = dmrg(H,psi0;nsweeps,maxdim,cutoff,outputlevel=1)

    EVs = expect(vac, "phi")
    push!(VEVs, sum(EVs)/length(EVs))
end

EEC_data = h5open("1d_EEC_data/CPEV,N=$N,a=$a,l=$l,m_i=$m_i,m_f=$m_f,len=$len", "w")
EEC_data["VEVs"] = VEVs
close(EEC_data)