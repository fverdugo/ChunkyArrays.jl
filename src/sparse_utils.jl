
"""
    indextype(a)

Return the element type of the vector
used to store the row or column indices in the sparse matrix `a`. 
"""
function indextype end

indextype(a::AbstractSparseMatrix) = indextype(typeof(a))
indextype(a::Type{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = Ti
indextype(a::Type{SparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti} = Ti

"""
    for (i,j,v) in nziterator(a)
    ...
    end

Iterate over the non zero entries of `a` returning the corresponding
row `i`, column `j` and value `v`.
"""
function nziterator end

nziterator(a::SparseArrays.AbstractSparseMatrixCSC) = NZIteratorCSC(a)

struct NZIteratorCSC{A}
    matrix::A
end

Base.length(a::NZIteratorCSC) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSC{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSC = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSC}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSC}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSC)
    if nnz(a.matrix) == 0
        return nothing
    end
    col = 0
    knext = nothing
    while knext === nothing
        col += 1
        ks = nzrange(a.matrix,col)
        knext = iterate(ks)
    end
    k, kstate = knext
    i = Int(rowvals(a.matrix)[k])
    j = col
    v = nonzeros(a.matrix)[k]
    (i,j,v), (col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC,state)
    col, kstate = state
    ks = nzrange(a.matrix,col)
    knext = iterate(ks,kstate)
    if knext === nothing
        while knext === nothing
            if col == size(a.matrix,2)
                return nothing
            end
            col += 1
            ks = nzrange(a.matrix,col)
            knext = iterate(ks)
        end
    end
    k, kstate = knext
    i = Int(rowvals(a.matrix)[k])
    j = col
    v = nonzeros(a.matrix)[k]
    (i,j,v), (col,kstate)
end

nziterator(a::SparseMatrixCSR) = NZIteratorCSR(a)

struct NZIteratorCSR{A}
    matrix::A
end

Base.length(a::NZIteratorCSR) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSR{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSR = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSR}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSR}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSR)
    if nnz(a.matrix) == 0
        return nothing
    end
    row = 0
    ptrs = a.matrix.rowptr
    knext = nothing
    while knext === nothing
        row += 1
        ks = nzrange(a.matrix,row)
        knext = iterate(ks)
    end
    k, kstate = knext
    i = row
    j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
    v = nonzeros(a.matrix)[k]
    (i,j,v), (row,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR,state)
    row, kstate = state
    ks = nzrange(a.matrix,row)
    knext = iterate(ks,kstate)
    if knext === nothing
        while knext === nothing
            if row == size(a.matrix,1)
                return nothing
            end
            row += 1
            ks = nzrange(a.matrix,row)
            knext = iterate(ks)
        end
    end
    k, kstate = knext
    i = row
    j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
    v = nonzeros(a.matrix)[k]
    (i,j,v), (row,kstate)
end

struct SubSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
    parent::A
    indices::B
    inv_indices::C
    function SubSparseMatrix(
            parent::AbstractSparseMatrix{T},
            indices::Tuple,
            inv_indices::Tuple) where T

        A = typeof(parent)
        B = typeof(indices)
        C = typeof(inv_indices)
        new{T,A,B,C}(parent,indices,inv_indices)
    end
end

Base.size(a::SubSparseMatrix) = map(length,a.indices)
Base.IndexStyle(::Type{<:SubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::SubSparseMatrix,i::Integer,j::Integer)
    I = a.indices[1][i]
    J = a.indices[2][j]
    a.parent[I,J]
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC} where T,
        B::AbstractVector,
        α::Number,
        β::Number)

    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    rv = rowvals(Ap)
    for (j,J) in enumerate(cols)
        αxj = B[j] * α
        for p in nzrange(Ap,J)
            I = rv[p]
            i = invrows[I]
            if i>0
                C[i] += nzv[p]*αxj
            end
        end
    end
    C
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::SubSparseMatrix{T,<:SparseMatrixCSR} where T,
        B::AbstractVector,
        α::Number,
        β::Number)

    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    cv = colvals(Ap)
    o = getoffset(Ap)
    for (i,I) in enumerate(rows)
        for p in nzrange(Ap,I)
            J = cv[p]+o
            j = invcols[J]
            if j>0
                C[i] += nzv[p]*B[j]*α
            end
        end
    end
    C
end

function LinearAlgebra.fillstored!(A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC},v) where T
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    rv = rowvals(Ap)
    for (j,J) in enumerate(cols)
        for p in nzrange(Ap,J)
            I = rv[p]
            i = invrows[I]
            if i>0
                nzv[p]=v
            end
        end
    end
    A
end

function LinearAlgebra.fillstored!(A::SubSparseMatrix{T,<:SparseMatrixCSR},v) where T
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    cv = colvals(Ap)
    o = getoffset(Ap)
    for (i,I) in enumerate(rows)
        for p in nzrange(Ap,I)
            J = cv[p]+o
            j = invcols[J]
            if j>0
                nzv[p] = v
            end
        end
    end
    A
end

"""
    nzindex(a,i,j)

Return the position in `nonzeros(a)` that stores the non zero value of `a` at row `i`
and column `j`.
"""
function nzindex end

function nzindex(A::SparseArrays.AbstractSparseMatrixCSC, i0::Integer, i1::Integer)
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    ptrs = SparseArrays.getcolptr(A)
    r1 = Int(ptrs[i1])
    r2 = Int(ptrs[i1+1]-1)
    (r1 > r2) && return -1
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? 0 : r1
end

function nzindex(A::SparseMatrixCSR, i0::Integer, i1::Integer)
  if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
  o = getoffset(A)
  Bi = getBi(A)
  r1 = Int(A.rowptr[i0]+o)
  r2 = Int(A.rowptr[i0+1]-Bi)
  (r1 > r2) && return -1
  i1o = i1-o
  k = searchsortedfirst(colvals(A), i1o, r1, r2, Base.Order.Forward)
  ((k > r2) || (colvals(A)[k] != i1o)) ? 0 : k
end

"""
    compresscoo(T,args...)

Like `sparse(args...)`, but generates a sparse matrix of type `T`.
"""
function compresscoo end

compresscoo(a::AbstractSparseMatrix,args...) = compresscoo(typeof(a),args...)

function compresscoo(
  ::Type{SparseMatrixCSC{Tv,Ti}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer,
  combine=+) where {Tv,Ti}

  sparse(
    EltypeVector(Ti,I),
    EltypeVector(Ti,J),
    EltypeVector(Tv,V),
    m,n,combine)
end

function compresscoo(
  ::Type{SparseMatrixCSR{Bi,Tv,Ti}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer,
  combine=+) where {Bi,Tv,Ti}

  sparsecsr(
    Val(Bi),
    EltypeVector(Ti,I),
    EltypeVector(Ti,J),
    EltypeVector(Tv,V),
    m,n,combine)
end

struct EltypeVector{T,V} <: AbstractVector{T}
  parent::V
  function EltypeVector(::Type{T},parent::V) where {T,V<:AbstractVector}
    new{T,V}(parent)
  end
end
EltypeVector(::Type{T},parent::AbstractVector{T}) where T = parent
Base.size(v::EltypeVector) = size(v.parent)
Base.axes(v::EltypeVector) = axes(v.parent)
Base.@propagate_inbounds Base.getindex(v::EltypeVector{T},i::Integer) where T = convert(T,v.parent[i])
Base.@propagate_inbounds Base.setindex!(v::EltypeVector,w,i::Integer) = (v.parent[i] = w)
Base.IndexStyle(::Type{<:EltypeVector{T,V}}) where {T,V} = IndexStyle(V)

struct SparseMatrixCOO{A,B,C,T} <: AbstractMatrix{T}
    I::A
    J::B
    V::C
    m::Int
    n::Int
    function SparseMatrixCOO(I,J,V,m,n)
        T = eltype(V)
        A = typeof(I)
        B = typeof(J)
        C = typeof(V)
        new{A,B,C,T}(I,J,V,m,n)
    end
end
Base.size(a::SparseMatrixCOO) = (a.m,a.n)
Base.IndexStyle(::Type{<:SparseMatrixCOO}) = IndexCartesian()
function Base.getindex(a::SparseMatrixCOO,i::Int,j::Int)
    v = zero(eltype(a))
    for p in 1:nnz(a)
        if a.I[p] == i && a.J[p] == j
            v += a.V[p]
        end
    end
    v
end
SparseArrays.nnz(a::SparseMatrixCOO) = length(a.V)
SparseArrays.findnz(a::SparseMatrixCOO) = (a.I,a.J,a.V)
indextype(a::SparseMatrixCOO) = eltype(a.I)
nziterator(a::SparseMatrixCOO) = zip(a.I,a.J,a.V)

function sparse_coo(I,J,V,m,n)
    @boundscheck begin
        @assert all(i->(i in 1:m),I)
        @assert all(j->(j in 1:n),J)
    end
    SparseMatrixCOO(I,J,V,m,n)
end

function sparse_coo!(A::SparseMatrixCOO,V)
    copy!(A.V,V)
    A
end

function similar_coo(
        coo,
        ::Type{Tv},
        ::Type{Ti},
        _size=size(coo),
        _nnz=nnz(coo)
    ) where {Ti,Tv}

    I,J,V = findnz(coo)
    m,n = _size
    SparseMatrixCOO(
               similar(I,Ti,_nnz),
               similar(J,Ti,_nnz),
               similar(V,Tv,_nnz),
               m,
               n)
end

function similar_coo(coo,_size=size(coo),_nnz=nnz(coo))
    I,J,V = findnz(coo)
    m,n = _size
    SparseMatrixCOO(
               similar(I,_nnz),
               similar(J,_nnz),
               similar(V,_nnz),
               m,
               n)
end

function to_csc(A)
    I,J,V = findnz(A)
    m,n = size(A)
    sparse(I,J,V,m,n)
end

function to_csc!(B::SparseArrays.AbstractSparseMatrixCSC,A)
    LinearAlgebra.fillstored!(B,0)
    nzvals_B = nonzeros(B)
    for (i,j,v) in nziterator(A)
        k = nzindex(B,i,j)
        nzvals_B[k] += v
    end
    B
end

function sparse_csc(I,J,V,m,n)
    A = sparse(I,J,V,m,n)
    K = zeros(eltype(I),length(I))
    for q in 1:length(K)
        i = I[q]
        j = J[q]
        k = nzindex(A,i,j)
        K[q] = k
    end
    (A,K)
end

function sparse_csc!(A,K,V)
    LinearAlgebra.fillstored!(A,0)
    A_nz = nonzeros(A)
    for (k,v) in zip(K,V)
        A_nz[k] += v
    end
    A
end

