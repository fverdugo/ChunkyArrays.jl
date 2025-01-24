module DebugArraySpMtMMTests

using PartitionedArrays
using SparseArrays

include(joinpath("..","primitives_tests.jl"))

M = sparse(1:5,1:5,1:5)
@test nnz(M-M) == nnz(M)
display(M-M)

with_debug(primitives_tests)

end # module
