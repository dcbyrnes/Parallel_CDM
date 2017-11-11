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
    var f_in = c.fopen("./data/INPUT_MATRIX", "rb")
    var f_out = c.fopen("./data/OUTPUT_DATA", "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var nrows : uint64 = dim[0] -- Dimension of output vector.
    var ncols : uint64 = dim[1] -- Dimension of feature vector.
    c.printf("n: %d m: %d \n", nrows, ncols)

    -- Create region Matrix (A).
    var A = region(ispace(int2d, { x = nrows, y = ncols }), float)
    var data : float[2]
    for i = 0, nrows do
        read_data(f_in, ncols, data)
        for j = 0, ncols do
            A[{i,j}] = data[j]
            c.printf("data: %f\n", A[{i,j}])
        end
    end
    var is_m = ispace(int1d, nrows)
    var is_n = ispace(int1d, ncols)

    -- Create output vector (y).
    var output_data : float[2]
    read_data(f_out, nrows, output_data)
    var y = region(is_m, float)
    for i = 0, nrows do
      y[i] = output_data[i] 
    end
    
    c.fclose(f_in)
    c.fclose(f_out)

    -- Use (parallel/distributed) coordinate descent to solve Ax=y.
    var x = region(is_n, float)
    var d = region(is_n, float)
    -- Initialize iterate vector x.
    for i in is_n do
        x[i] = 10
        d[i] = 1
    end

    var eta : float = 1
    var b : float = 1.0
    var lam : float = 0.5
    hl_grad(is_n, d, x, b, -1, lam, eta)
    print(is_n, d) 
end

regentlib.start(toplevel)
