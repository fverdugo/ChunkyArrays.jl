module DebugArraySpMtMMTests

using PartitionedArrays
using Test

include(joinpath("..","spmtmm_tests.jl"))

v = 1:5
M = sparse(v,v,v)
@test nnz(M-M) == nnz(M)
display(M-M)

M = sparsecsr(v,v,v)
@test nnz(M-M) == nnz(M)
display(M-M)

with_debug(spmtmm_tests)

end # module
