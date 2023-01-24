using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances

function p_vector_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    rows = prange(uniform_partition,rank,(2,2),(6,6))

    a1 = pvector(rows)
    a2 = pvector(inds->zeros(Int,length(inds)),rows)
    a3 = PVector{OwnAndGhostValues{Vector{Int}}}(undef,rows)
    for a in [a1,a2,a3]
        b = similar(a)
        b = similar(a,Int)
        b = similar(a,Int,rows)
        b = similar(typeof(a),rows)
        copy!(b,a)
        b = copy(a)
        fill!(b,5.)
        @test length(a) == length(rows)
        @test a.rows === rows
        @test b.rows === rows
    end

    a = pfill(4,rows)
    a = pzeros(rows)
    a = pones(rows)
    a = prand(rows)
    a = prandn(rows)
    assemble!(a) |> wait
    consistent!(a) |> wait

    @test a == copy(a)

    n = 10
    I,V = map(rank) do rank
        Random.seed!(rank)
        I = rand(1:n,5)
        Random.seed!(2*rank)
        V = rand(1:2,5)
        I,V
    end |> unpack

    rows = prange(uniform_partition,rank,n)
    a = pvector!(I,V,rows) |> fetch

    @test any(i->i>n,a) == false
    @test all(i->i<n,a)
    @test minimum(a) <= maximum(a)

    b = 2*a
    b = a*2
    b = a/2
    c = a .+ a
    c = a .+ b .+ a
    c = a - b
    c = a + b

    r = reduce(+,a)
    @test sum(a) == r
    @test norm(a) > 0
    @test sqrt(a⋅a) ≈ norm(a)
    @test euclidean(a,a) + 1 ≈ 1

    n = 10
    parts = rank
    indices = map(parts) do part
        if part == 1
            LocalIndices(n,part,[1,2,3,5,7,8],Int32[1,1,1,2,3,3])
        elseif part == 2
            LocalIndices(n,part,[2,4,5,10],Int32[1,2,2,4])
        elseif part == 3
            LocalIndices(n,part,[6,7,8,5,4,10],Int32[3,3,3,2,2,4])
        else
            LocalIndices(n,part,[1,3,7,9,10],Int32[1,1,3,4,4])
        end
    end
    rows = PRange(indices)
    v = pzeros(rows)
    map(parts,v.values,v.rows.indices) do part, values, indices
        local_to_owner = get_local_to_owner(indices)
        for lid in 1:length(local_to_owner)
            owner = local_to_owner[lid]
            if owner == part
                values[lid] = 10*part
            end
        end
    end
    consistent!(v) |> wait

    map(v.values,v.rows.indices) do values, indices
        local_to_owner = get_local_to_owner(indices)
        for lid in 1:length(local_to_owner)
            owner = local_to_owner[lid]
            @test values[lid] == 10*owner
        end
    end

    map(get_local_values(v)) do values
        fill!(values,10.0)
    end

    assemble!(v) |> wait
    map(parts,get_local_values(v)) do part,values
        if part == 1
            @test values == [20.0, 20.0, 20.0, 0.0, 0.0, 0.0]
        elseif part == 2
            @test values == [0.0, 20.0, 30.0, 0.0]
        elseif part == 3
            @test values == [10.0, 30.0, 20.0, 0.0, 0.0, 0.0]
        else
            @test values == [0.0, 0.0, 0.0, 10.0, 30.0]
        end
    end

    n = 10

    gids = map(parts) do part
        if part == 1
            [1,4,6]
        elseif part == 2
            [3,1,2,8]
        elseif part == 3
            [1,9,6]
        else
            [3,2,8,10]
        end
    end
    rows = prange(uniform_partition,parts,n)
    values = map(copy,gids)
    v = pvector!(gids,values,rows) |> fetch
    u = 2*v
    map(u.values,v.values) do u,v
        @test u == 2*v
    end
    u = v + u
    map(u.values,v.values) do u,v
        @test u == 3*v
    end
    @test any(i->i>4,v) == true
    @test any(i->i>17,v) == false
    @test all(i->i<17,v) == true
    @test all(i->i<4,v) == false
    @test maximum(v) == 16
    @test minimum(v) == 0
    @test maximum(i->i-1,v) == 15
    @test minimum(i->i-1,v) == -1

    w = copy(v)
    rmul!(w,-1)
    @test all(i->i==0,v+w)

    @test w == w
    @test w != v

    @test sqeuclidean(w,v) ≈ (norm(w-v))^2
    @test euclidean(w,v) ≈ norm(w-v)

    w = similar(v)
    w = zero(v)
    @test isa(w,PVector)
    @test norm(w) == 0
    @test sum(w) == 0

    w = v .- u
    @test isa(w,PVector)
    w =  1 .+ v
    @test isa(w,PVector)
    w =  v .+ 1
    @test isa(w,PVector)
    w =  v .+ w .- u
    @test isa(w,PVector)
    w =  v .+ 1 .- u
    @test isa(w,PVector)

    w .= v .- u
    w .= v .- 1 .- u
    w .= u
    map(w.values,u.values) do w,u
        @test w == u
    end

    w = v .- u
    @test isa(w,PVector)
    w =  v .+ w .- u
    @test isa(w,PVector)
    w =  v .+ 1 .- u
    @test isa(w,PVector)

    w .= v .- u
    w .= v .- 1 .- u
    w .= u
    map(get_own_values(w),get_own_values(u)) do w,u
        @test w == u
    end

end
