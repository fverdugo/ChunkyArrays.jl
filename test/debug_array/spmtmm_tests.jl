module DebugArraySpMtMMTests

using PartitionedArrays
using Test

include(joinpath("..","spmtmm_tests.jl"))

v = 1:5
M = sparse(v,v,v)
Z = subtract(M,M)
@test nnz(Z) == nnz(M)
display(Z)

M = sparsecsr(v,v,v)
Z = subtract(M,M)
@test nnz(Z) == nnz(M)
display(Z)

with_debug(spmtmm_tests)

end # module
