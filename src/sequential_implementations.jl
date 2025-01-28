function Base.:*(A::SparseMatrixCSR{Bi,TvA,TiA},B::SparseMatrixCSR{Bi,TvB,TiB}) where {Bi,TvA,TiA,TvB,TiB}
    C = ascsc(B)*ascsc(A)
    ascsr(C)
end

function Base.:*(At::Transpose{Tv, SparseMatrixCSR{Bi,Tv,Ti}},B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    C = ascsc(B)*transpose(ascsc(At.parent))
    ascsr(C)
end

function Base.:*(A::SparseMatrixCSR{Bi,Tv,Ti},Bt::Transpose{Tv, SparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti}
    C = transpose(ascsc(Bt.parent))*ascsc(A)
    ascsr(C)
end

function Base.:*(At::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},Bt::Transpose{Tv, SparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti}
    C = transpose(ascsc(Bt.parent))*transpose(ascsc(At.parent))
    ascsr(C)
end

function Base.:*(x::Number,A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(A.colval), map(a -> x*a, A.nzval))
end
function Base.:*(A::SparseMatrixCSR,x::Number) *(x,A) end

function Base.:/(A::SparseMatrixCSR{Bi,Tv,Ti},x::Number) where {Bi,Tv,Ti}
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(A.colval), map(a -> a/x, A.nzval))
end


# Alternative to lazy csr to csc for matrix addition that does not drop structural zeros.
function Base.:+(A::SparseMatrixCSR{Bi,Tv,Ti},B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    IC = Vector{Ti}(undef, p+1)
    JC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    for i in 1:p
        IC[i] = pC
        jpA_range = nzrange(A, i)
        jpA, jpA_end = jpA_range.start, jpA_range.stop
        jpB_range = nzrange(B, i)
        jpB, jpB_end = jpB_range.start, jpB_range.stop
        while jpA <= jpA_end && jpB <= jpB_end
            jA = JA[jpA]
            jB = JB[jpB]
            if jA < jB
                JC[pC] = jA
                VC[pC] = VA[jpA]
                jpA += 1
            elseif jB < jA
                JC[pC] = jB
                VC[pC] = VB[jpB]
                jpB += 1
            else
                JC[pC] = jA
                VC[pC] = VA[jpA] + VB[jpB]
                jpA += 1
                jpB += 1
            end
            pC += 1
        end
        while jpA <= jpA_end
            JC[pC] = JA[jpA]
            VC[pC] = VA[jpA]
            jpA += 1
            pC += 1
        end
        while jpB <= jpB_end
            JC[pC] = JB[jpB]
            VC[pC] = VB[jpB]
            jpB += 1
            pC += 1
        end
    end
    IC[end] = pC
    resize!(JC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSR{Bi}(p,q,IC,JC,VC)   # A += B
end

# Alternative to lazy csr to csc for matrix subtraction that does not drop structural zeros. Subtracts B from A, i.e. A - B.
function Base.:-(A::SparseMatrixCSR{Bi,Tv,Ti},B::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    nnz_C_upperbound = nnz(A) + nnz(B)
    p,r = size(A)
    IC = Vector{Ti}(undef, p+1)
    JC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    for i in 1:p
        IC[i] = pC
        jpA_range = nzrange(A, i)
        jpA, jpA_end = jpA_range.start, jpA_range.stop
        jpB_range = nzrange(B, i)
        jpB, jpB_end = jpB_range.start, jpB_range.stop
        while jpA <= jpA_end && jpB <= jpB_end
            jA = JA[jpA]
            jB = JB[jpB]
            if jA < jB
                JC[pC] = jA
                VC[pC] = VA[jpA]
                jpA += 1
            elseif jB < jA
                JC[pC] = jB
                VC[pC] = -VB[jpB]
                jpB += 1
            else
                JC[pC] = jA
                VC[pC] = VA[jpA] - VB[jpB]
                jpA += 1
                jpB += 1
            end
            pC += 1
        end
        while jpA <= jpA_end
            JC[pC] = JA[jpA]
            VC[pC] = VA[jpA]
            jpA += 1
            pC += 1
        end
        while jpB <= jpB_end
            JC[pC] = JB[jpB]
            VC[pC] = -VB[jpB]
            jpB += 1
            pC += 1
        end
    end
    IC[end] = pC
    resize!(JC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSR{Bi}(p,r,IC,JC,VC)   # A += B
end

function Base.:-(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(A.colval), map(a->-a, A.nzval))
end

# Alternative to lazy csr to csc for matrix addition that does not drop structural zeros.
function Base.:+(A::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    if size(A) != size(B) && throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    JC = Vector{Ti}(undef, q+1)
    IC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    IA = rowvals(A)
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:q
        JC[j] = pC
        ipA_range = nzrange(A, j)
        ipA, ipA_end = ipA_range.start, ipA_range.stop
        ipB_range = nzrange(B, j)
        ipB, ipB_end = ipB_range.start, ipB_range.stop
        while ipA <= ipA_end && ipB <= ipB_end
            iA = IA[ipA]
            iB = IB[ipB]
            if iA < iB
                IC[pC] = iA
                VC[pC] = VA[ipA]
                ipA += 1
            elseif iB < iA
                IC[pC] = iB
                VC[pC] = VB[ipB]
                ipB += 1
            else
                IC[pC] = iA
                VC[pC] = VA[ipA] + VB[ipB]
                ipA += 1
                ipB += 1
            end
            pC += 1
        end
        while ipA <= ipA_end
            IC[pC] = IA[ipA]
            VC[pC] = VA[ipA]
            ipA += 1
            pC += 1
        end
        while ipB <= ipB_end
            IC[pC] = IB[ipB]
            VC[pC] = VB[ipB]
            ipB += 1
            pC += 1
        end
    end
    JC[end] = pC
    resize!(IC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSC{Tv,Ti}(p,q,JC,IC,VC)
end

# Alternative to lazy csr to csc for matrix subtraction that does not drop structural zeros. Subtracts B from A, i.e. A - B.
function Base.:-(A::SparseMatrixCSC{Tv,Ti},B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    if size(A) == size(B) || throw(DimensionMismatch("Size of B $(size(B)) must match size of A $(size(A))"));end
    p,q = size(A)
    nnz_C_upperbound = nnz(A) + nnz(B)
    JC = Vector{Ti}(undef, q+1)
    IC = Vector{Ti}(undef, nnz_C_upperbound)
    VC = Vector{Tv}(undef, nnz_C_upperbound)
    
    pC = 1
    IA = rowvals(A)
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:q
        JC[j] = pC
        ipA_range = nzrange(A, j)
        ipA, ipA_end = ipA_range.start, ipA_range.stop
        ipB_range = nzrange(B, j)
        ipB, ipB_end = ipB_range.start, ipB_range.stop
        while ipA <= ipA_end && ipB <= ipB_end
            iA = IA[ipA]
            iB = IB[ipB]
            if iA < iB
                IC[pC] = iA
                VC[pC] = VA[ipA]
                ipA += 1
            elseif iB < iA
                IC[pC] = iB
                VC[pC] = VB[ipB]
                ipB += 1
            else
                IC[pC] = iA
                VC[pC] = VA[ipA] - VB[ipB]
                ipA += 1
                ipB += 1
            end
            pC += 1
        end
        while ipA <= ipA_end
            IC[pC] = IA[ipA]
            VC[pC] = VA[ipA]
            ipA += 1
            pC += 1
        end
        while ipB <= ipB_end
            IC[pC] = IB[ipB]
            VC[pC] = -VB[ipB]
            ipB += 1
            pC += 1
        end
    end
    JC[end] = pC
    resize!(IC, (pC-1))
    resize!(VC, (pC-1))
    SparseMatrixCSC{Tv,Ti}(p,q,JC,IC,VC)
end

function Base.:-(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    SparseMatrixCSC{Tv,Ti}(size(A)..., copy(A.colptr), copy(A.rowval), map(a->-a, A.nzval))
end


function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    B::SparseMatrixCSC{Tv,Ti},
    cache) where {Tv,Ti}
    mul!(ascsr(C),ascsr(B),ascsr(A),cache)
end


function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    B::SparseMatrixCSC{Tv,Ti},
    α::Number,
    β::Number,
    cache) where {Tv,Ti}
    mul!(ascsr(C),ascsr(B),ascsr(A),α,β,cache)
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                            B::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .= 0
    IC = rowvals(C)
    JA = rowvals(A) # When virtually transposed rowvals represent colvals.
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:s
        # loop over columns "j" in row i of A
        Bj = nzrange(B, j)
        ptrB_start = Bj.start
        ptrB_stop = Bj.stop
        for ip in nzrange(C, j)
            i = IC[ip]
            # loop over columns "k" in row j of B
            Ai = nzrange(A, i)
            ptrB = ptrB_start
            ptrA = Ai.start
            vC = 0
            while ptrA <= Ai.stop && ptrB <= ptrB_stop
                jA = JA[ptrA]
                iB = IB[ptrB]
                if jA < iB
                    ptrA += 1
                elseif iB < jA
                    ptrB += 1
                else # jA == iB
                    vC += VA[ptrA]*VB[ptrB]
                    ptrA += 1
                    ptrB += 1
                end
            end
            VC[ip] = vC
        end
    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                            B::SparseMatrixCSC{Tv,Ti},
                            α::Number,
                            β::Number) where {Tv,Ti}
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    IC = rowvals(C)
    VC .*= β
    JA = rowvals(A) # When virtually transposed rowvals represent colvals.
    VA = nonzeros(A)
    IB = rowvals(B)
    VB = nonzeros(B)
    for j in 1:s
        # loop over columns "j" in row i of A
        Bj = nzrange(B, j)
        for jp in nzrange(C, j)
            i = IC[jp]
            # loop over columns "k" in row j of B
            Ai = nzrange(A, i)
            ptrB = Bj.start
            ptrA = Ai.start
            vC = 0
            while ptrA <= Ai.stop && ptrB <= Bj.stop
                jA = JA[ptrA]
                iB = IB[ptrB]
                if jA == iB
                    vC += VA[ptrA]*VB[ptrB]
                    ptrA += 1
                    ptrB += 1
                elseif jA < iB
                    ptrA += 1
                else
                    ptrB += 1
                end
            end
            VC[jp] += α*vC
        end
    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    Bt::Transpose{Tv,SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
    mul!(ascsr(C),transpose(ascsr(B)),ascsr(A))
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            A::SparseMatrixCSR{Bi,Tv,Ti},
                            B::SparseMatrixCSR{Bi,Tv,Ti},
                            cache) where {Bi,Tv,Ti}
    a,b = size(C)
    p,q = size(A)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(Tv)
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    # A cache here would remove need for allocating acumulating arrays
    # xb = zeros(Ti, p)
    xb,x = cache
    xb .= 0
    # x = similar(xb, Tv) # sparse accumulator, can be zeros() to remove if statement in inner loop.
    for i in 1:p # !
        # loop over rows Ai in col Bj
        for jpa in nzrange(A, i) 
            ja = JA[jpa]
            va = VA[jpa]
            # loop over columns "k" in row j of B
            for jpb in nzrange(B, ja) 
                jb = JB[jpb]
                vb = VB[jpb]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[jb] != i
                    xb[jb] = i
                    x[jb] = va*vb
                else
                    x[jb] += va*vb
                end
            end
        end
        for jpc in nzrange(C,i)
            jc = JC[jpc]
            # To support in-place products whose sparsity patterns are subsets of the sparsity of C, this check is required.
            if xb[jc] == i
                VC[jpc] = x[jc]
            end
        end
    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    Bt::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
    cache) where {Tv,Ti}
    mul!(ascsr(C),transpose(ascsr(B)),ascsr(A),cache)
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            A::SparseMatrixCSR{Bi,Tv,Ti},
                            B::SparseMatrixCSR{Bi,Tv,Ti},
                            α::Number,
                            β::Number,
                            cache) where {Bi,Tv,Ti}
    a,b = size(C)
    p,q = size(A)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    JA = colvals(A)
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    # A cache here would remove need for allocating acumulating arrays
    # xb = zeros(Ti, p)
    xb,x = cache
    xb .= 0
    # x = similar(xb, Tv) # sparse accumulator, can be zeros() to remove if statement in inner loop.
    for i in 1:p # !
        # loop over rows Ai in col Bj
        for jpa in nzrange(A, i) 
            ja = JA[jpa]
            va = VA[jpa]
            # loop over columns "k" in row j of B
            for jpb in nzrange(B, ja) 
                jb = JB[jpb]
                vb = VB[jpb]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[jb] != i
                    xb[jb] = i
                    x[jb] = va*vb
                else
                    x[jb] += va*vb
                end
            end
        end
        for jpc in nzrange(C,i)
            jc = JC[jpc]
            # To support in-place products whose sparsity patterns are subsets of the sparsity of C, this check is required.
            if xb[jc] == i
                VC[jpc] += α * x[jc]
            end
        end
    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
                            A::SparseMatrixCSC{Tv,Ti},
                            Bt::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                            α::Number,
                            β::Number,
                            cache) where {Tv,Ti}
    mul!(ascsr(C),transpose(ascsr(Bt.parent)),ascsr(A),α,β,cache)
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                            B::SparseMatrixCSC{Tv,Ti},
                            cache) where {Tv,Ti}
    mul!(ascsr(C),ascsr(B),transpose(ascsr(At.parent)))
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
                            B::SparseMatrixCSC{Tv,Ti},
                            α::Number,
                            β::Number,
                            cache) where {Tv,Ti}
    mul!(ascsr(C),ascsr(A),transpose(ascsr(At.parent)),α,β)
end

# Workaround to supply in-place mul! with auxiliary array, as these are not returned by multiply function exported by SparseArrays
function construct_spmm_cache(A::SparseMatrixCSR{Bi,Tv,Ti} where Bi) where {Tv,Ti}
    q = size(A,2)
    xb = zeros(Ti,q)
    x = similar(xb,Tv)
    xb,x
end
function construct_spmm_cache(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    construct_spmm_cache(ascsr(A))
end

function construct_spmtm_cache(A::SparseMatrixCSR{Bi,Tv,Ti} where Bi) where {Tv,Ti}
    q = size(A,2)
    xb = zeros(Ti,q)
    x = similar(xb,Tv)
    xb,x
end

function construct_spmtm_cache(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    construct_spmtm_cache(ascsr(A))
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
                            B::SparseMatrixCSR{Bi,Tv,Ti},
                            cache) where {Bi,Tv,Ti}
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .= zero(Tv)
    JC = colvals(C)
    JA = colvals(A) # When virtually transposed colvals represent rowvals.
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    xb,x = cache
    xb .= 0
    for k in 1:q
        # loop over columns "j" in row i of B
        for jpb in nzrange(B,k)
            jb = JB[jpb]
            vb = VB[jpb]
            xb[jb] = k
            x[jb] = vb
        end
        for ipa in nzrange(A,k)
            ia = JA[ipa] # interpret column index of A as row index of A^T.
            va = VA[ipa]
            for jpc in nzrange(C, ia)
                jc = JC[jpc]
                # This check is required, as the outerproduct might not contribute to to all nonzero entries in this row of C.
                if xb[jc] == k
                    VC[jpc] += va*x[jc]
                end
            end
        end

    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            At::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
                            B::SparseMatrixCSR{Bi,Tv,Ti},
                            α::Number,
                            β::Number,
                            cache) where {Bi,Tv,Ti}
    a,b = size(C)
    p,q = size(At)
    r,s = size(B)
    if q != r && throw(DimensionMismatch("A has dimensions ($(p),$(q)) but B has dimensions ($(p),$(q))"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("C has dimensions $((a,b)) but AB will have dimensions ($(p),$(s))"));end
    A = At.parent
    VC = nonzeros(C)
    VC .*= β
    JC = colvals(C)
    JA = colvals(A) # When virtually transposed colvals represent rowvals.
    VA = nonzeros(A)
    JB = colvals(B)
    VB = nonzeros(B)
    xb,x = cache
    xb .= 0
    for k in 1:q
        # loop over columns "j" in row i of B
        for jpb in nzrange(B,k)
            jb = JB[jpb]
            vb = VB[jpb]
            xb[jb] = k
            x[jb] = α*vb
        end
        for ipa in nzrange(A,k)
            ia = JA[ipa] # interpret column index of A as row index of A^T.
            va = VA[ipa]
            for jpc in nzrange(C, ia)
                jc = JC[jpc]
                # This check is required, as the outerproduct might not contribute to to all nonzero entries in this row of C.
                if xb[jc] == k
                    VC[jpc] += va*x[jc]
                end
            end
        end

    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            A::SparseMatrixCSR{Bi,Tv,Ti},
                            Bt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti}
    mul!(ascsc(C), transpose(ascsc(Bt.parent)), ascsc(A))
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSR{Bi,Tv,Ti},
                            A::SparseMatrixCSR{Bi,Tv,Ti},
                            Bt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
                            α::Number,
                            β::Number) where {Bi,Tv,Ti}
    mul!(ascsc(C), transpose(ascsc(Bt.parent)), ascsc(A), α, β)
    C
end

# PtAP variants
function rap(Plt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}}, A::SparseMatrixCSR{Bi,Tv,Ti}, Pr::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    p,q = size(Plt)
    m,r = size(A)
    n,s = size(Pr)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    function rap_symbolic_count!(R,A,Pr)
        JR = R.data
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        xbRA = zeros(Ti, r)
        xbC = zeros(Ti, s) # this vector will also serve as as colptr array in halfperm
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rPr = find_max_row_length(Pr)

        max_rC = max((max_rR*max_rA*max_rPr),(max_rA*max_rR))
        JRA = Vector{Ti}(undef,max_rC)
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        JAP = Vector{Ti}(undef,min(max_rA*max_rPr,s)) # upper bound estimate for length of virtual row of AP
        xbRA .= 0
        xbC .= 0
        cache = (xbRA,JRA,xbC,JAP)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC), cache # values not yet initialized
    end
    function rap_symbolic_fill!(C,R,A,Pr,cache)
        (xbRA,JRA,xbC,JAP) = cache
        JC = colvals(C)
        JR = R.data
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        pC = 0
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1 
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        pC += 1
                        xbC[k] = i
                        JC[pC] = k
                    end
                end
            end
        end
        xbC .= 0
        outer_cache = (xbC,similar(xbC, Tv),JAP)
        C, outer_cache # values not yet initialized
    end
    function _rap(Plt,A,Pr)
        R = symbolic_halfperm(Plt.parent)
        C,symbolic_cache = rap_symbolic_count!(R,A,Pr) # precompute nz structure with a symbolic transpose
        _,outer_cache = rap_symbolic_fill!(C,R,A,Pr,symbolic_cache)
        Ct = symbolic_halfperm(C)
        symbolic_halfperm!(C,Ct)
        rap!(C,Plt,A,Pr,outer_cache),(outer_cache...,R)
    end
    _rap(Plt,A,Pr)
end

function rap(Plt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
             A::SparseMatrixCSR{Bi,Tv,Ti},
             Pr::SparseMatrixCSR{Bi,Tv,Ti},
             cache) where {Bi,Tv,Ti}
    p,q = size(Plt)
    m,r = size(A)
    n,s = size(Pr)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    
    function rap_symbolic_count!(R,A,Pr)
        JR = R.data
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        xbRA = zeros(Ti, r)
        xbC = zeros(Ti, s) # this vector will also serve as as colptr array in halfperm
        max_rR = find_max_row_length(R)
        max_rA = find_max_row_length(A)
        max_rPr = find_max_row_length(Pr)

        max_rC = max((max_rR*max_rA*max_rPr),(max_rA*max_rR))
        JRA = Vector{Ti}(undef,max_rC)
        IC = Vector{Ti}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{Ti}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        JAP = Vector{Ti}(undef,min(max_rA*max_rPr,s)) # upper bound estimate for length of virtual row of AP
        xbRA .= 0
        xbC .= 0
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC),(xbRA,JRA,xbC,JAP) # values in CSR matrix not yet initialized
    end
    function rap_symbolic_fill!(C,R,A,Pr,cache)
        (xbRA,JRA,xbC,JAP) = cache
        JC = colvals(C)
        JR = R.data
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        pC = 0
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in jagged_range(R, i)
                j = JR[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1 
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        pC += 1
                        xbC[k] = i
                        JC[pC] = k
                    end
                end
            end
        end
        xbC .= 0
        C, (xbC,similar(xbC, Tv),JAP) # values not yet initialized
    end
    function _rap(Plt,A,Pr,old_cache)
        xb,x,JAP,R = old_cache
        old_outer_cache = (xb,x,JAP)
        C,symbolic_cache = rap_symbolic_count!(R, A, Pr)
        _,new_outer_cache = rap_symbolic_fill!(C,R, A, Pr, symbolic_cache)
        Ct = symbolic_halfperm(C)
        symbolic_halfperm!(C,Ct)
        outer_cache = map((c1,c2) -> length(c1) >= length(c2) ? c1 : c2, old_outer_cache,new_outer_cache)
        rap!(C,Plt,A,Pr,outer_cache),(outer_cache...,R)
    end
    _rap(Plt,A,Pr,cache)
end

function reduce_spmtmm_cache(cache,::Type{SparseMatrixCSR})
    (xb,x,JAP,_) = cache
    (xb,x,JAP)
end

function rap!(C::SparseMatrixCSR{Bi,Tv,Ti}, 
              Plt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
              A::SparseMatrixCSR{Bi,Tv,Ti},
              Pr::SparseMatrixCSR{Bi,Tv,Ti},
              cache) where {Bi,Tv,Ti}
    (a,b) = size(C)
    p,q = size(Plt)
    m,r = size(A)
    n,s = size(Pr)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("Dimensions of C $(size(C)) don't match dimensions of R*A*P ($p,$q)*($m,$r)*($n,$s)."));end
    Pl = Plt.parent
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(Tv)

    JA = colvals(A)
    VA = nonzeros(A)
    JPr = colvals(Pr)
    VPr = nonzeros(Pr)
    xb, x, JAP = cache
    xb .= 0
    # loop over rows in A
    for i in 1:m
        lp = 0
        # loop over columns "j" in row i of A
        for jp in nzrange(A, i)
            j = JA[jp]
            va = VA[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(Pr, j)
                k = JPr[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    lp += 1
                    JAP[lp] = k
                    xb[k] = i
                    x[k] = va * VPr[kp]
                else
                    x[k] += va * VPr[kp]
                end
            end
        end
        for kp in nzrange(Pl, i)
            k = colvals(Pl)[kp] # rowvals when transposed conceptually
            v = nonzeros(Pl)[kp]
            for jp in nzrange(C,k)
                j = JC[jp]
                if xb[j] == i
                    VC[jp] += v*x[j]
                end 
            end
        end
    end
    C
end

function rap!(C::SparseMatrixCSR{Bi,Tv,Ti},
              Plt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}},
              A::SparseMatrixCSR{Bi,Tv,Ti},
              Pr::SparseMatrixCSR{Bi,Tv,Ti},
              α::Number,
              β::Number,
              cache) where {Bi,Tv,Ti}
    (a,b) = size(C)
    p,q = size(Plt)
    m,r = size(A)
    n,s = size(Pr)
    if r != n && throw(DimensionMismatch("Invalid dimensions for A*P: ($m,$r)*($n,$s),"));end
    if q != m && throw(DimensionMismatch("Invalid dimensions: R*AP: ($p,$q)*($m,$s)"));end
    if (a,b) != (p,s) && throw(DimensionMismatch("Dimensions of C $(size(C)) don't match dimensions of R*A*P ($p,$q)*($m,$r)*($n,$s)."));end
    Pl = Plt.parent
    JC = colvals(C)
    VC = nonzeros(C)
    JA = colvals(A)
    VA = nonzeros(A)
    JPr = colvals(Pr)
    VPr = nonzeros(Pr)
    xb, x, JAP = cache
    xb .= 0
    VC .*= β
    # loop over rows in A
    for i in 1:m
        lp = 0
        # loop over columns "j" in row i of A
        for jp in nzrange(A, i)
            j = JA[jp]
            va = α*VA[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(Pr, j)
                k = JPr[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    lp += 1
                    JAP[lp] = k
                    xb[k] = i
                    x[k] = va*VPr[kp]
                else
                    x[k] += va*VPr[kp]
                end
            end
        end
        for kp in nzrange(Pl, i)
            k = colvals(Pl)[kp] # rowvals when transposed conceptually
            vpl = nonzeros(Pl)[kp]
            for jp in nzrange(C,k)
                j = JC[jp]
                if xb[j] == i
                    VC[jp] += vpl*x[j]
                end 
            end
        end
    end
    C
end

# rap variants
function rap(Pl::SparseMatrixCSR{Bi,Tv,TiPl},
             A::SparseMatrixCSR{Bi,Tv,TiA},
             Pr::SparseMatrixCSR{Bi,Tv,TiPr}) where {Bi,Tv,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Pr)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    function rap_symbolic!(Pl,A,Pr)
        JPl = colvals(Pl)
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        xbRA = zeros(TiA, r)
        xbC = zeros(TiA, s+1) # this vector will also serve as as colptr array in halfperm
        xRA = similar(xbRA, Tv) # sparse accumulator
        xC = similar(xbC, Tv) # sparse accumulator
        max_rPl = find_max_row_length(Pl)
        max_rA = find_max_row_length(A)
        max_rPr = find_max_row_length(Pr)

        max_rC = max((max_rPl*max_rA*max_rPr),(max_rA*max_rPl))
        JRA = Vector{TiA}(undef,max_rC)
        IC = Vector{TiA}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0
            # loop over columns "j" in row i of A
            for jp in nzrange(Pl, i)
                j = JPl[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{TiA}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        cache = (xbRA,xRA,JRA,xbC,xC)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC), cache # values not yet initialized
    end
    function rap_numeric!(C,Pl,A,Pr,cache)
        JPl = colvals(Pl)
        VPl = nonzeros(Pl)
        JA = colvals(A)
        VA = nonzeros(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        VPr = nonzeros(Pr)
        JC = colvals(C)
        VC = nonzeros(C)
        (xbRA,xRA,JRA,xbC,xC) = cache
        jpC = 1
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(Pl, i)
                j = JPl[jp]
                vpl = VPl[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                        xRA[k] = vpl * VA[kp]
                    else
                        xRA[k] += vpl * VA[kp]
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        JC[jpC] = k
                        jpC += 1
                        xC[k] = xRA[j]*VPr[kp]
                    else
                        xC[k] += xRA[j]*VPr[kp]
                    end
                end
            end
            for ind in nzrange(C,i)
                j = JC[ind]
                VC[ind] = xC[j]
            end
        end
    end
    function _rap(Pl,A,Pr)
        C,(xbRA,xRA,JRA,xbC,xC) = rap_symbolic!(Pl,A,Pr)
        xbRA .= 0
        xbC .= 0
        cache = (xbRA,xRA,JRA,xbC,xC)
        rap_numeric!(C,Pl,A,Pr,cache)
        Ct = halfperm!(xbC,similar(colvals(C)),similar(nonzeros(C)),C)
        halfperm!(C,Ct)
        C,cache
    end
    _rap(Pl,A,Pr)
end

# Reuses internal arrays of A!!!
function construct_spmmm_cache(C::SparseMatrixCSR,A::SparseMatrixCSR)
    cache = JaggedArray(colvals(A), A.rowptr)
end

function construct_spmmm_cache(C::SparseMatrixCSC,A::SparseMatrixCSC)
    cache = JaggedArray(rowvals(A), A.colptr)
end

function reduce_spmtmm_cache(cache,::Type{M} where M <: SparseMatrixCSR)
    (xb,x,JAP,_) = cache
    (xb,x,JAP)
end

function reduce_spmtmm_cache(cache,::Type{M}  where M <: SparseMatrixCSC)
    reduce_spmmmt_cache(cache,SparseMatrixCSR)
end

function rap(Pl::SparseMatrixCSR{Bi,Tv,TiPl},
             A::SparseMatrixCSR{Bi,Tv,TiA},
             Pr::SparseMatrixCSR{Bi,Tv,TiPr},
             cache) where {Bi,Tv,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Pr)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    function rap_symbolic!(Pl,A,Pr,cache)
        JPl = colvals(Pl)
        JA = colvals(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        (xbRA,_,JRA,xbC,_) = cache
        IC = Vector{TiA}(undef,p+1)
        nnz_C = 1
        IC[1] = nnz_C
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(Pl, i)
                j = JPl[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                    end
                end
            end
            ccC = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        ccC += 1
                    end
                end
            end
            nnz_C += ccC
            IC[i+1] = nnz_C
        end
        JC = Vector{TiA}(undef, nnz_C-1)
        VC = zeros(Tv,nnz_C-1)
        SparseMatrixCSR{Bi}(p,s,IC,JC,VC) # values not yet initialized
    end
    function rap_numeric!(C,Pl,A,Pr,cache)
        JPl = colvals(Pl)
        VPl = nonzeros(Pl)
        JA = colvals(A)
        VA = nonzeros(A)
        JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
        VPr = nonzeros(Pr)
        JC = colvals(C)
        VC = nonzeros(C)
        (xbRA,xRA,JRA,xbC,xC) = cache
        jpC = 1
        for i in 1:p
            ccRA = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
            # loop over columns "j" in row i of A
            for jp in nzrange(Pl, i)
                j = JPl[jp]
                vpl = VPl[jp]
                # loop over columns "k" in row j of B
                for kp in nzrange(A, j)
                    k = JA[kp]
                    # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                    if xbRA[k] != i
                        ccRA += 1
                        JRA[ccRA] = k
                        xbRA[k] = i
                        xRA[k] = vpl * VA[kp]
                    else
                        xRA[k] += vpl * VA[kp]
                    end
                end
            end
            for jp in 1:ccRA
                j = JRA[jp]
                for kp in nzrange(Pr,j)
                    k = JPr[kp]
                    if xbC[k] != i
                        xbC[k] = i
                        JC[jpC] = k
                        jpC += 1
                        xC[k] = xRA[j]*VPr[kp]
                    else
                        xC[k] += xRA[j]*VPr[kp]
                    end
                end
            end
            for ind in nzrange(C,i)
                j = JC[ind]
                VC[ind] = xC[j]
            end
        end
    end
    function _rap(Pl,A,Pr,old_cache)
        max_rPl = find_max_row_length(Pl)
        max_rA = find_max_row_length(A)
        max_rPr = find_max_row_length(Pr)
        (xbRA,xRA,JRA,xbC,xC) = old_cache
        max_rC = max((max_rPl*max_rA*max_rPr),(max_rA*max_rPl))
        JRA2 = max_rC > length(JRA) ? similar(JRA,max_rC) : JRA
        if r > length(xbRA)
            xbRA2 = similar(xbRA,r)
            xRA2 = similar(xRA,r)
        else
            xbRA2 = xbRA
            xRA2 = xRA
        end

        new_cache = (xbRA2,xRA2,JRA2,xbC,xC)
        xbRA2 .= 0
        xbC .= 0
        C = rap_symbolic!(Pl,A,Pr,new_cache)
        xbRA2 .= 0
        xbC .= 0
        rap_numeric!(C,Pl,A,Pr,new_cache)
        Ct = halfperm!(xbC,similar(colvals(C)),similar(nonzeros(C)),C)
        halfperm!(C,Ct)
        C,new_cache
    end
    _rap(Pl,A,Pr,cache)
end

function reduce_spmmmt_cache(cache,::Type{M} where M <: SparseMatrixCSR)
    (xbRA,xRA,JRA,_,_) = cache
    (xbRA,xRA,JRA)
end

function reduce_spmmmt_cache(cache,::Type{M} where M <: SparseMatrixCSC)
    reduce_spmtmm_cache(cache,SparseMatrixCSR)
end

function rap!(C::SparseMatrixCSR{Bi,Tv,TiC},
              Pl::SparseMatrixCSR{Bi,Tv,TiPl},
              A::SparseMatrixCSR{Bi,Tv,TiA},
              Pr::SparseMatrixCSR{Bi,Tv,TiPr},
              cache) where {Bi,Tv,TiC,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Pr)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    JPl = colvals(Pl)
    VPl = nonzeros(Pl)
    JA = colvals(A)
    VA = nonzeros(A)
    JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
    VPr = nonzeros(Pr)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .= zero(Tv)
    (xbRA,xRA,JRA,xbC,xC) = cache
    xbRA .= 0
    xbC .= 0
    for i in 1:p
        lp = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
        # loop over columns "j" in row i of A
        for jp in nzrange(Pl, i)
            j = JPl[jp]
            vpl = VPl[jp]

            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                va = VA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xbRA[k] != i
                    lp += 1
                    JRA[lp] = k
                    xbRA[k] = i
                    xRA[k] = vpl * va
                else
                    xRA[k] += vpl * va
                end
            end
        end
        for jp in 1:lp
            j = JRA[jp]
            vra = xRA[j]
            for kp in nzrange(Pr,j)
                k = JPr[kp]
                if xbC[k] != i
                    xbC[k] = i
                    xC[k] = vra*VPr[kp]
                else
                    xC[k] += vra*VPr[kp]
                end
            end
        end
        for ind in nzrange(C,i)
            j = JC[ind]
            if xbC[j] == i
                VC[ind] = xC[j]
            end
        end
    end
    C
end

function rap!(C::SparseMatrixCSR{Bi,Tv,TiC},
              Pl::SparseMatrixCSR{Bi,Tv,TiPl},
              A::SparseMatrixCSR{Bi,Tv,TiA},
              Pr::SparseMatrixCSR{Bi,Tv,TiPr},
              α::Number,
              β::Number,
              cache) where {Bi,Tv,TiC,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Pr)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    JPl = colvals(Pl)
    VPl = nonzeros(Pl)
    JA = colvals(A)
    VA = nonzeros(A)
    JPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
    VPr = nonzeros(Pr)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    (xbRA,xRA,JRA,xbC,xC) = cache
    xbRA .= 0
    xbC .= 0
    xC .= zero(Tv)
    for i in 1:p
        lp = 0 # local column pointer, refresh every row, start at 0 to allow empty rows
        # loop over columns "j" in row i of A
        for jp in nzrange(Pl, i)
            j = JPl[jp]
            vpl = VPl[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xbRA[k] != i
                    lp += 1
                    JRA[lp] = k
                    xbRA[k] = i
                    xRA[k] = vpl * VA[kp]
                else
                    xRA[k] += vpl * VA[kp]
                end
            end
        end
        for jp in 1:lp
            j = JRA[jp]
            for kp in nzrange(Pr,j)
                k = JPr[kp]
                if xbC[k] != i
                    xbC[k] = i
                    xC[k] = xRA[j]*VPr[kp]
                else
                    xC[k] += xRA[j]*VPr[kp]
                end
            end
        end
        for ind in nzrange(C,i)
            j = JC[ind]
            if xbC[j] == i
                VC[ind] += α*xC[j]
            end
        end
    end
    C
end

# RARt variants
function rap(Pl::SparseMatrixCSR{Bi,Tv,TiA},
             A::SparseMatrixCSR{Bi,Tv,TiB},
             Prt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,TiC}}) where {Bi,Tv,TiA,TiB,TiC}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Prt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions for RA*P: ($p,$r)*($n,$s)"));end
    rap(Pl,A,copy(Prt))
end

function rap(Pl::SparseMatrixCSR{Bi,Tv,TiA},
             A::SparseMatrixCSR{Bi,Tv,TiB},
             Prt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,TiC}},cache) where {Bi,Tv,TiA,TiB,TiC}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Prt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions for RA*P: ($p,$r)*($n,$s)"));end
    rap(Pl,A,copy(Prt),cache)
end

function rap!(C::SparseMatrixCSR{Bi,Tv,TiC},
              Pl::SparseMatrixCSR{Bi,Tv,TiPl}, 
              A::SparseMatrixCSR{Bi,Tv,TiA}, 
              Prt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,TiPr}},
              cache) where {Bi,Tv,TiC,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Prt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    Pr = Prt.parent
    JPl = colvals(Pl)
    VPl = nonzeros(Pl)
    JA = colvals(A)
    VA = nonzeros(A)
    IPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
    VPr = nonzeros(Pr)
    JC = colvals(C)
    VC = nonzeros(C)
    # some cache items are present with the regular rap product in mind, which is how the allocating verison is performed
    xb,x = cache
    xb .= 0
    for i in 1:p
        # loop over columns "j" in row i of A
        for jp in nzrange(Pl, i)
            j = JPl[jp]
            vpl = VPl[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    xb[k] = i
                    x[k] = vpl * VA[kp]
                else
                    x[k] += vpl * VA[kp]
                end
            end
        end
        for jpPr in nzrange(C,i)
            jPr = JC[jpPr]
            v = Tv(0)
            for ip in nzrange(Pr,jPr)
                iPr = IPr[ip]
                if xb[iPr] == i
                    v += x[iPr]*VPr[ip]
                end
            end
            VC[jpPr] = v
        end
    end
    C
end

function rap!(C::SparseMatrixCSR{Bi,Tv,TiC},
              Pl::SparseMatrixCSR{Bi,Tv,TiPl},
              A::SparseMatrixCSR{Bi,Tv,TiA},
              Prt::Transpose{Tv,SparseMatrixCSR{Bi,Tv,TiPr}},
              α::Number,
              β::Number,
              cache) where {Bi,Tv,TiC,TiPl,TiA,TiPr}
    p,q = size(Pl)
    m,r = size(A)
    n,s = size(Prt)
    if q == m || throw(DimensionMismatch("Invalid dimensions for R*A: ($p,$q)*($m,$r),"));end
    if r == n || throw(DimensionMismatch("Invalid dimensions: RA*P: ($p,$r)*($n,$s)"));end
    Pr = Prt.parent
    JPl = colvals(Pl)
    VPl = nonzeros(Pl)
    JA = colvals(A)
    VA = nonzeros(A)
    IPr = colvals(Pr) # colvals can be interpreted as rowvals when Pr is virtually transposed.
    VPr = nonzeros(Pr)
    JC = colvals(C)
    VC = nonzeros(C)
    VC .*= β
    # some cache items are present with the regular rap product in mind, which is how the allocating verison is performed
    xb,x = cache
    xb .= 0
    for i in 1:p
        # loop over columns "j" in row i of A
        for jp in nzrange(Pl, i)
            j = JPl[jp]
            vpl = VPl[jp]
            # loop over columns "k" in row j of B
            for kp in nzrange(A, j)
                k = JA[kp]
                # since C is constructed rowwise, xb tracks if a column index is present in a new row in C.
                if xb[k] != i
                    xb[k] = i
                    x[k] = vpl * VA[kp]
                else
                    x[k] += vpl * VA[kp]
                end
            end
        end
        for jpPr in nzrange(C,i)
            jPr = JC[jpPr]
            v = Tv(0)
            for ip in nzrange(Pr,jPr)
                iPr = IPr[ip]
                if xb[iPr] == i
                    v += x[iPr]*VPr[ip]
                end
            end
            VC[jpPr] += α*v
        end
    end
    C
end

### CSC in terms of CSR
function rap(A::SparseMatrixCSC{Tv,TiA},
             B::SparseMatrixCSC{Tv,TiB},
             C::SparseMatrixCSC{Tv,TiC}) where {Tv,TiA,TiB,TiC}
    D,cache = rap(ascsr(C),ascsr(B),ascsr(A))
    ascsc(D),cache
end

function rap(A::SparseMatrixCSC{Tv,TiA},
             B::SparseMatrixCSC{Tv,TiB},
             C::SparseMatrixCSC{Tv,TiC},
             cache) where {Tv,TiA,TiB,TiC}
    D,new_cache = rap(ascsr(C),ascsr(B),ascsr(A),cache)
    ascsc(D),new_cache
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::SparseMatrixCSC{Tv,TiA},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              cache) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),cache)
    D
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::SparseMatrixCSC{Tv,TiA},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              cache::JaggedArray{X,Y} where {X<:Integer, Y<:Integer},
              acc) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),cache,acc)
    D
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::SparseMatrixCSC{Tv,TiA},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              α::Number,
              β::Number,
              cache) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),α,β,cache)
    D
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::SparseMatrixCSC{Tv,TiA},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              α::Number,
              β::Number,
              cache::JaggedArray{X,Y} where {X <: Integer, Y<:Integer},
              acc) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),ascsr(A),α,β,cache,acc)
    D
end

# PtAP
function rap(A::Transpose{Tv,SparseMatrixCSC{Tv,TiA}},
             B::SparseMatrixCSC{Tv,TiB},
             C::SparseMatrixCSC{Tv,TiC}) where {Tv,TiA,TiB,TiC}
    D,cache = rap(ascsr(C),ascsr(B),transpose(ascsr(A.parent)))
    ascsc(D),cache
end

function rap(A::Transpose{Tv,SparseMatrixCSC{Tv,TiA}},
             B::SparseMatrixCSC{Tv,TiB},
             C::SparseMatrixCSC{Tv,TiC},
             cache) where {Tv,TiA,TiB,TiC}
    D,cache = rap(ascsr(C),ascsr(B),transpose(ascsr(A.parent)),cache)
    ascsc(D),cache
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::Transpose{Tv,SparseMatrixCSC{Tv,TiA}},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              cache) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),transpose(ascsr(A.parent)),cache)
    D
end

function rap!(D::SparseMatrixCSC{Tv,TiD},
              A::Transpose{Tv,SparseMatrixCSC{Tv,TiA}},
              B::SparseMatrixCSC{Tv,TiB},
              C::SparseMatrixCSC{Tv,TiC},
              α::Number,
              β::Number,
              cache) where {Tv,TiD,TiA,TiB,TiC}
    rap!(ascsr(D),ascsr(C),ascsr(B),transpose(ascsr(A.parent)),α,β,cache)
    D
end

# RARt
function rap(A::SparseMatrixCSC{Tv,Ti},
             B::SparseMatrixCSC{Tv,Ti},
             C::Transpose{Tv,SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti<:Integer}
    D,new_cache = rap(transpose(ascsr(C.parent)),ascsr(B),ascsr(A))
    ascsc(D),new_cache
end
function rap(A::SparseMatrixCSC{Tv,Ti},
             B::SparseMatrixCSC{Tv,Ti},
             C::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
             cache) where {Tv,Ti<:Integer}
    D,new_cache = rap(transpose(ascsr(C.parent)),ascsr(B),ascsr(A),cache)
    ascsc(D),new_cache
end

function rap!(D::SparseMatrixCSC{Tv,Ti},
              A::SparseMatrixCSC{Tv,Ti},
              B::SparseMatrixCSC{Tv,Ti},
              C::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
              cache) where {Tv,Ti<:Integer}
    rap!(ascsr(D),transpose(ascsr(C.parent)),ascsr(B),ascsr(A),cache)
    D
end

function rap!(D::SparseMatrixCSC{Tv,Ti},
              A::SparseMatrixCSC{Tv,Ti},
              B::SparseMatrixCSC{Tv,Ti},
              C::Transpose{Tv,SparseMatrixCSC{Tv,Ti}},
              α::Number,
              β::Number,
              cache) where {Tv,Ti<:Integer}
    rap!(ascsr(D),transpose(ascsr(C.parent)),ascsr(B),ascsr(A),α,β,cache)
    D
end