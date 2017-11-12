import "regent"

-- C APIs
local c = regentlib.c

terra get_dimensions(f : &c.FILE, d : &uint32)
    c.fscanf(f, "%d %d \n", &d[0], &d[1])
end

terra read_data(f : &c.FILE, n : uint64, input : &float)
    var x : float, y : float
    for  i = 0, n do
        c.fscanf(f, "%f", &input[i]) 
    end
    c.fscanf(f, "\n")
end

-- Helper funtion to print vectors.
task print(is : ispace(int1d), 
           x : region(ispace(int1d), float))
    where 
        reads(x)
    do
    for i in is do
        c.printf("%f \n", x[i])
    end
end

-- Sets y = a * x + y, where a is a scalar and x, y are vectors. 
task saxpy(is : ispace(int1d), 
           x  : region(ispace(int1d), float), 
           y  : region(ispace(int1d), float), 
           a  : float)
where
  reads(x, y), writes(y)
do 
  __demand(__vectorize)
  for i in is do
    y[i] += a*x[i]
  end
end

-- Returns evaluation of linear function f(x) = w^T x + b.
task f(is : ispace(int1d),
       w  : region(ispace(int1d), float),
       x  : region(ispace(int1d), float),
       b  : float)
where 
    reads(x, w)
do
    var f_ : float = 0.0
    --__demand(__vectorize)
    for i in is do
        f_ += x[i] * w[i]
    end
    return (f_ + b)
end

-- Gradient of hinge loss function.
task hl_grad(is  : ispace(int1d),
             w   : region(ispace(int1d), float),
             x   : region(ispace(int1d), float), 
             b   : double,
             y   : int32,
             lam : float,
             eta : float)
where 
    reads(x), 
    reads writes (w)
do
    var z = region(is, float)
    fill(z, 0.0)
    -- z += (-eta * lam) * w_t.
    saxpy(is, w, z, -eta * lam)
    if (y * f(is,w,x,b) < 1) then
        var z_ = region(is, float)
        fill(z_, 0.0)
        -- z_ += -eta * y_i * x_i
        saxpy(is, x, z_, -y * eta)
        -- z += z
        saxpy(is, z_, z, 1)
    end
    -- w += -z
    saxpy(is, z, w, -1)
end

task toplevel()
    -- Load data from file.
    var f_in = c.fopen("./data/toy_examples/data_a.txt", "rb")
    --var f_in = c.fopen("./data/toy_examples/INPUT_MATRIX", "rb")
    var f_out = c.fopen("./data/toy_examples/OUTPUT_DATA", "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var nrows : uint64 = dim[0] -- Dimension of output vector.
    var ncols : uint64 = dim[1] -- Dimension of feature vector.
    var is_m = ispace(int1d, nrows)
    var is_n = ispace(int1d, ncols) -- Bias term included.
    c.printf("n: %d m: %d \n", nrows, ncols)

    -- Create region Matrix (A) and response vector (y).
    var A = region(ispace(int2d, { x = nrows, y = (ncols-1) }), float)
    var y = region(is_m, float)
    var data : float[2]
    for i = 0, nrows do
        read_data(f_in, ncols, data)
        for j = 0, ncols do
            if j == 0 then
                y[i] = data[j]
                c.printf("%f --->   ", y[i])
            else
                A[{i,j-1}] = data[j]
                c.printf("%f ", A[{i,j-1}])
            end
        end
        c.printf("\n")
    end
--[[
    -- Create output vector (y).
    var output_data : float[2]
    read_data(f_out, nrows, output_data)
    var y = region(is_m, float)
    for i in is_m do
      --y[i] = output_data[i] 
    end
--]]
    c.fclose(f_in)
    c.fclose(f_out)

    -- Minimize the (hinge) loss function.
    var w = region(is_n, float)
    -- Initialize weight vector (w).
    for i in is_n do
        -- TODO(DB) remove this hack and cast the proper way.
        w[i] = c.rand() / (c.RAND_MAX-1.3)
    end

    var eta : float = 1
    var b : float = 0
    var lam : float = 0.005

    -- TODO(db) Iterate over partition of A.
    var rs = ispace(int2d, { x = nrows, y = 1 })
    var parx = partition(equal, A, rs)
    var x = region(is_n, float)
    for r = 0, nrows do
        -- TODO(db) Another hack.
        x[0] = 1.0
        for i = 0, ncols-1 do
            x[i+1] = A[{r, i}]
            --c.printf("H: %d %f \n", i, x[i+1])
        end
        --print(is_n, x)
        -- Use the hinge loss to compute weight vector (w).
        hl_grad(is_n, w, x, b, y[r], lam, eta)
        --print(is_n, w) 
    end
    print(is_n, w) 
    
    var converged = false
    --while not converged do

    --end
end

regentlib.start(toplevel)
