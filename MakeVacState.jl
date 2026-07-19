using ITensors, ITensorMPS, LinearAlgebra, HDF5
ITensors.disable_threaded_blocksparse() #Disables block sparse multithreading
BLAS.set_num_threads(4) #Enables multithreading with BLAS

### Initial Parameters ###
N = parse(Int, ARGS[1]) #Number of lattice sites per dimension
d = 1 #Number of spatial dimensions
Dim = parse(Int, ARGS[2]) #Truncated local Hilbert space dimension
a = parse(Float64, ARGS[3]) #Lattice spacing

n_0 = round(Int, ((N-1)/2)) #Index of the point at the center of the lattice (0-based indexing).

mass = parse(Float64, ARGS[4])
m0 = parse(Float64, ARGS[5]) #Basis frequency
l = parse(Float64, ARGS[6]) #phi^4 coupling strength

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

sites = siteinds("Boson", N; dim=Dim) #Create ITensor sites

H = MPO(H_OS, sites) #Hamiltonian MPO

### Vacuum State MPS ###
psi0 = random_mps(sites;linkdims=10)
nsweeps = parse(Int, ARGS[7])
maxdim = [50, 50, 100, 100, 200, 200]
cutoff = [1E-10]
energy, vac = dmrg(H,psi0;nsweeps,maxdim,cutoff)

EEC_MPS = h5open("1d_EEC_MPS/N=$N,a=$a,dim=$Dim,l=$l,m=$mass", "w")
write(EEC_MPS, "sites", sites)
write(EEC_MPS, "vac", vac)
close(EEC_MPS)

@show linkdims(vac)