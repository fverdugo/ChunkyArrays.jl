
struct LinearSolverStatus <: AbstractType
    steps::Int
    steps_since_update::Int
    updates::Int
end

function LinearSolverStatus()
    LinearSolverStatus(0,0,0)
end

function update(status::LinearSolverStatus)
    steps = status.steps
    steps_since_update = 0
    updates = status.updates + 1
    LinearSolverStatus(steps,steps_since_update,updates)
end

function step(status::LinearSolverStatus)
    steps = status.steps + 1
    steps_since_update = status.steps_since_update + 1
    updates = status.updates
    LinearSolverStatus(steps,steps_since_update,updates)
end

function Base.show(io::IO,k::MIME"text/plain",data::LinearSolverStatus)
    println("Linear solver updates since creation: $(data.updates)")
    println("Linear solver steps since creation: $(data.steps)")
    println("Linear solver steps since last update: $(data.steps_since_update)")
end

function status(a)
    workspace(a).status
end

struct Convergence{T}
    current::T
    target::T
    iteration::Int
    iterations::Int
end

function Convergence(iterations,target)
    current = zero(typeof(target))
    iteration = 0
    Convergence(current,target,iteration,iterations)
end

function print_progress(a::Convergence,verbosity)
    s =verbosity.indentation
    @printf "%s%6i %6i %12.3e %12.3e\n" s a.iteration a.iterations a.current a.target
end

function converged(a::Convergence)
    a.current <= a.target
end

function tired(a::Convergence)
    a.iteration >= a.iterations
end

function start(a::Convergence,current)
    target = a.target
    iteration = 0
    iterations = a.iterations
    Convergence(current,target,iteration,iterations)
end

function step(a::Convergence,current)
    target = a.target
    iteration = a.iteration + 1
    iterations = a.iterations
    Convergence(current,target,iteration,iterations)
end

function Base.show(io::IO,k::MIME"text/plain",data::Convergence)
    if converged(data)
        println("Converged in $(data.iteration) iterations")
    else
        println("Not converged in $(data.iteration) iterations")
    end
end

function convergence(a)
    workspace(a).convergence
end

function workspace(a)
    a.workspace
end

function timer_output(a)
    workspace(a).timer_output
end

abstract type AbstractLinearSolver <: AbstractType end

function solve!(x,P::AbstractLinearSolver,f::AbstractVector;zero_guess=false)
    next = step!(x,P,f;zero_guess)
    while next !== nothing
        x,P,state = next
        next = step!(x,P,f,state)
    end
    x,P
end

function LinearAlgebra.ldiv!(x,P::AbstractLinearSolver,b)
    if uses_initial_guess(x)
        fill!(x,zero(eltype(x)))
    end
    solve!(x,P,b;zero_guess=true)
    x
end

uses_initial_guess(P) = true

function matrix(a::AbstractMatrix)
    a
end

function verbosity(;indentation="")
    (;indentation)
end

struct LinearAlgebra_LU{A} <: AbstractLinearSolver
    workspace::A
end

function LinearAlgebra_lu(A;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput(),
    )
    @timeit timer_output "LinearAlgebra_lu" begin
        factors = lu(A)
        status = LinearSolverStatus()
        workspace = (;factors,verbose,verbosity,timer_output,status)
        LinearAlgebra_LU(workspace)
    end
end

function LinearAlgebra_lu(x,p;kwargs...)
    A = matrix(p)
    LinearAlgebra_lu(A;kwargs...)
end

function update!(P::LinearAlgebra_LU,p)
    (;factors,verbose,verbosity,timer_output,status) = P.workspace
    @timeit timer_output "LinearAlgebra_lu update!" begin
        A = matrix(p)
        lu!(factors,A)
        status = update(status)
        workspace = (;factors,verbose,verbosity,timer_output,status)
        LinearAlgebra_LU(workspace)
    end
end

uses_initial_guess(P::LinearAlgebra_LU) = false

function step!(x,P::LinearAlgebra_LU,b,state=:start;kwargs...)
    (;factors,verbose,verbosity,timer_output,status) = P.workspace
    if state === :stop
        return nothing
    end
    @timeit timer_output "LinearAlgebra_lu step!" begin
        ldiv!(x,factors,b)
        status = step(status)
        state = :stop
        workspace = (;factors,verbose,verbosity,timer_output,status)
        P = LinearAlgebra_LU(workspace)
        x,P,state
    end
end

struct JacobiCorrection{A} <: AbstractLinearSolver
    workspace::A
end

function jacobi_correction(A,b;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())
    @timeit timer_output "jacobi_correction" begin
        Adiag = dense_diag(A)
        status = LinearSolverStatus()
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        JacobiCorrection(workspace)
    end
end

function diagonal_solve!(x,Adiag,b)
    x .= Adiag .\ b
    x
end

function update!(P::JacobiCorrection,p)
    (;Adiag,verbose,verbosity,timer_output,status) = P.workspace
    @timeit timer_output "jacobi_correction update!" begin
        A = matrix(p)
        dense_diag!(Adiag,A)
        status = update(status)
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        JacobiCorrection(workspace)
    end
end

function step!(x,P::JacobiCorrection,b,state=:start;kwargs...)
    (;Adiag,verbose,verbosity,timer_output,status) = P.workspace
    if state === :stop
        return nothing
    end
    @timeit timer_output "jacobi_correction step!" begin
        diagonal_solve!(x,Adiag,b)
        status = step(status)
        state = :stop
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        P = JacobiCorrection(workspace)
        x,P,state
    end
end

struct IdentitySolver{A} <: AbstractLinearSolver
    workspace::A
end

function identity_solver(;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())
    @timeit timer_output "identity_solver" begin
        status = LinearSolverStatus()
        workspace = (;status,verbosity,verbose,timer_output)
        IdentitySolver(workspace)
    end
end

function update!(P::IdentitySolver,p)
    (;status,verbosity,verbose,timer_output) = P.workspace
    @timeit timer_output "identity_solver update!" begin
        status = update(status)
        workspace = (;status,verbosity,verbose,timer_output)
        IdentitySolver(workspace)
    end
end

function step!(x,P::IdentitySolver,b,state=:start;kwargs...)
    (;status,verbosity,verbose,timer_output) = P.workspace
        if state === :stop
            return nothing
        end
    @timeit timer_output "identity_solver step!" begin
        copy!(x,b)
        status = step(status)
        state = :stop
        workspace = (;status,verbosity,verbose,timer_output)
        P = IdentitySolver(workspace)
        x,P,state
    end
end

struct Richardson{A} <: AbstractLinearSolver
    workspace::A
end

function richardson(x,A,b;
        omega = 1,
        iterations = 10,
        preconditioner=identity_solver(),
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput()
    )
    @timeit timer_output "richardson" begin
        P = preconditioner
        r = similar(b)
        dx = similar(x,axes(A,2))
        status = LinearSolverStatus()
        target = 0
        convergence = Convergence(iterations,target)
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        Richardson(workspace)
    end
end

function update!(S::Richardson,p)
    (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output) = S.workspace
    @timeit timer_output "richardson update!" begin
        P = update!(P,p)
        status = update(status)
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        Richardson(workspace)
    end
end

function step!(x,S::Richardson,b,phase=:start;zero_guess=false,kwargs...)
    (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output) = S.workspace
    @timeit timer_output "richardson step!" begin
        @assert phase in (:start,:stop,:advance)
        if phase === :stop
            return nothing
        end
        if phase === :start
            current = convergence.iterations
            convergence = start(convergence,current)
            verbose && print_progress(convergence,verbosity)
            phase = :advance
        end
        dx .= x
        if zero_guess
            r .= .- b
        else
            @timeit timer_output "richardson step! mul!" begin
                mul!(r,A,dx)
            end
            r .-= b
        end
        @timeit timer_output "richardson step! ldiv!" begin
            ldiv!(dx,P,r)
        end
        x .-= omega .* dx
        current = convergence.current - 1
        convergence = step(convergence,current)
        status = step(status)
        verbose && print_progress(convergence,verbosity)
        if converged(convergence)
            phase = :stop
        end
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        S = Richardson(workspace)
        x,S,phase
    end
end

function jacobi(x,A,b;timer_output=TimerOutput(),iterations=10,kwargs...)
    preconditioner = jacobi_correction(A,b;timer_output,kwargs...)
    richardson(x,A,b;timer_output,preconditioner,iterations,kwargs...)
end

abstract type AbstractNonlinearSolver <: AbstractType end

function solve!(x,P::AbstractNonlinearSolver,f::AbstractNonlinearProblem;zero_guess=false)
    next = step!(x,P,f;zero_guess)
    while next !== nothing
        x,P,f,state = next
        next = step!(x,P,f,state)
    end
    x,P,f
end

struct NewtonRaphson{A} <: AbstractNonlinearSolver
    workspace::A
end

function linear_solver(a::NewtonRaphson)
    a.workspace.linear_solver
end

function newton_raphson(x,p;
        iterations=1000,
        reltol_residual=1e-12,
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput(),
        linear_solver = LinearAlgebra_lu(jacobian(update!(p,x));timer_output),
    )
    @timeit timer_output "newton_raphson" begin
        t = tangent(p)
        dx = similar(rhs(t),axes(matrix(t),2))
        target = zero(eltype(dx))
        convergence = Convergence(iterations,target)
        workspace = (;dx,linear_solver,verbose,verbosity,timer_output,convergence,reltol_residual)
        NewtonRaphson(workspace)
    end
end

function step!(x,S::NewtonRaphson,p,phase=:start;kwargs...)
    if phase === :stop
        return nothing
    end
    (;dx,linear_solver,verbose,verbosity,timer_output,convergence,reltol_residual) = S.workspace
    @timeit timer_output "newton_raphson step!" begin
        @assert phase in (:start,:stop,:advance)
        if phase === :start
            phase = :advance
            p = update!(p,x)
            r = residual(p)
            current = norm(r)
            target = reltol_residual*current
            convergence = start(convergence,target)
            verbose && print_progress(convergence,verbosity)
        end
        t = tangent(p)
        linear_solver = update!(linear_solver,matrix(t))
        dx,linear_solver = solve!(dx,linear_solver,rhs(t))
        x .-= dx
        p = update!(p,x)
        r = residual(p)
        current = norm(r)
        convergence = step(convergence,current)
        verbose && print_progress(convergence,verbosity)
        if converged(convergence) || tired(convergence)
            phase = :stop
        end
        workspace = (;dx,linear_solver,verbose,verbosity,timer_output,convergence,reltol_residual)
        S = NewtonRaphson(workspace)
        x,S,p,phase
    end
end

#abstract type AbstractODESolver <: AbstractType end
#
#struct ForwardEuler{A} <: AbstractODESolver
#    workspace::A
#end
#
#function forward_euler((u0,v),p)
#    function problem(t,u0)
#        function rj!(r,j,u)
#            v .= (u .- u0) ./ dt
#            p = update!(p,(u,v))
#
#
#            r,j
#        end
#    end
#
#
#end
#
#function step!(t,(u0,v),S::ForwardEuler,p,phase=:start;kwargs...)
#    p = problem(t)
#    u,S = solve!(u,S,g)
#
#
#    v .= (u .- u0) ./ dt
#    u0 .= u
#    S = ForwardEuler(workspace)
#
#    t+dt,(u0,v),S,p
#end


