import "regent"

-- C APIs
local c = regentlib.c
local sqrt = regentlib.sqrt(double)
local cmath = terralib.includec("math.h")
local PI = cmath.M_PI

-- Convergence tolerance. 
TOL = 1e-8
MAX_ITERS = 2

-- If z is a vector then add elements with z.i = ...   
fspace Vector {
    i : double
}

terra get_dimensions(f : &c.FILE, d : &uint32)
    c.fscanf(f, "%d %d \n", &d[0], &d[1])
end

terra read_data(f : &c.FILE, n : uint64, input : &double)
    var x : double, y : double
    for  i = 0, n do
        c.fscanf(f, "%lf", &input[i]) 
    end
    c.fscanf(f, "\n")
end

-- Returns the L2 norm of a Vector fspace.
task norm(x : region(Vector))
    where
        reads(x.i)
    do
    var sum : double = 0
    for e in x do
        sum += cmath.pow(e.i, 2)
    end
   return sqrt(sum) 
end

-- Scales the entries in a vector region by some constant.
task scale(x     : region(Vector),
           alpha : double)
where 
    reads writes (x.i)
do
    for e in x do
        e.i *= alpha
    end
end 

-- Returns evaluation of linear function f(x) = w^T x + b.
task f(x : region(Vector),
       w : region(Vector),
       b : double)
where 
    reads(x.i, w.i)
do
    var f_ : double = 0.0
    for e in x do
        f_ + x[e].i + w[e].i
    end
    return (f_ + b)
end

-- Gradient of hinge loss function.
task hl_grad(w   : region(Vector),
             x   : region(Vector), 
             b   : double,
             y   : uint32,
             eta : double)
where 
    reads(x), 
    reads writes (w)
do
    var grad : double = 0.0
    if (y * f(w,x,b) < 1) then
        grad = scale(w, eta) - scale(y,x)
    else
        grad = scale(w, eta)
    end 
end


-- Gradient of square loss function.
task square_loss_grad(n : uint32,
                      m : uint32,
                      x : region(Vector),
                      y : region(Vector),
                      A : region(ispace(int2d, { x = n, y = m }), double),
                      d : region(Vector))
where 
    reads(x, y, A, d), writes(d)
do  
    var inner_prod : double = 0
    for k = 0, m do
        for e in x do
            inner_prod += A[{k, e}] * x[e].i
            c.printf("A_ij: %f \n", A[{k,e}])
            c.printf("e_i: %f \n", x[e].i)
            c.printf("inner prod: %f \n", inner_prod)
        end 
        -- Compute gradient of loss. 
        var alpha : double = (inner_prod - y[k].i)
        for e in d do
            d[e].i = alpha * A[{k, e}]
            c.printf("A: %f \n", A[{k,e}])
        end 
    end
end

-- Update rule for gradient descent. 
-- Assumes the size of x and d are equal.
task update_rule(x             : region(Vector),
                 d             : region(Vector),
                 learning_rate : double)
    where 
        reads(d,x), writes(x)
    do
    for e in x do
        x[e].i -= learning_rate * d[e].i
    end
end

-- Helper funtion to print vectors.
task print(x : region(Vector))
    where 
        reads(x)
    do
    for e in x do
        c.printf("%f \n", e.i)
    end
end

-- Uses parallel coordinate descent method to minimize 
-- differentiable convex functions. 
task toplevel()
    -- Load data from file.
    var f_in = c.fopen("./project/Parallel_CDM/data/INPUT_MATRIX", "rb")
    var f_out = c.fopen("./project/Parallel_CDM/data/OUTPUT_DATA", "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var nrows : uint64 = dim[0] -- Dimension of output vector.
    var ncols : uint64 = dim[1] -- Dimension of feature vector.
    c.printf("n: %d m: %d \n", nrows, ncols)

    -- Create region Matrix (A).
    var A = region(ispace(int2d, { x = nrows, y = ncols }), double)
    var data : double[2]
    for i = 0, nrows do
        read_data(f_in, ncols, data)
        for j = 0, ncols do
            A[{i,j}] = data[j]
            c.printf("data: %f\n", A[{i,j}])
        end
    end
    
    -- Create output vector (y).
    var output_data : double[2]
    read_data(f_out, nrows, output_data)
    var y = region(ispace(ptr, nrows), Vector)
    for j = 0, nrows do
       y[j].i = output_data[j] 
    end

    -- Use (parallel/distributed) coordinate descent to solve Ax=y.
    var x = region(ispace(ptr, ncols), Vector)
    var d = region(ispace(ptr, ncols), Vector)
    -- Initialize iterate vector x.
    for e in x do
        e.i = 0
        d[e].i = 0
    end
    c.printf("Printing...\n")
    print(x)
    
    var delta : double = 1000
    var learning_rate : float = 0.05
    var counter : uint64 = 0
    --while del > TOL do
    while counter < MAX_ITERS do
        -- Compute search direction vector d.
        square_loss_grad(ncols, nrows, x, y, A, d)
    
        -- Iterator update rule.
        print(d)
        update_rule(x, d, learning_rate)
        
        --del = Vector2d{ y.x - x.x, y.y - x.y }:norm()
        --x = y 
        counter += 1
    end
    
    -- Print results.
    --c.printf("Iterations: %d \n", counter)
    c.printf("Done...\n")
    print(x)
    --c.printf("%f \n", norm(x))
    --c.printf("Minimized objective: %f \n", func(y))

end

regentlib.start(toplevel)
