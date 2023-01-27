module PartitionedArraysRunTests

using Test

@testset "jagged_array" begin include("jagged_array_tests.jl") end
@testset "sparse_utils" begin include("sparse_utils_tests.jl") end
@testset "debug_data" begin include("debug_data/runtests.jl") end
@testset "mpi_data" begin include("mpi_data/runtests.jl") end

end # module
