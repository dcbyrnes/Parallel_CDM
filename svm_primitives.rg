import "regent"

-- Helper module to parse data stored in libsvm format. 
local DataFile = require("libsvm_format_parser")

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
sqrt = regentlib.sqrt(double)
    
MAX_ITERS = 10

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

-- Copy the contents of row of 2D region to a 1D region (dimensions must agree).
-- x : 1D region to fill.
-- A : 2D region to copy from.
-- row : specific row of 2D region to copy from. 
task copy_row(x   : region(ispace(int1d), float),
              A   : region(ispace(int2d), float),
              row : uint64)
where
    reads(x, A), writes(x)
do
    __demand(__vectorize)
    for j in x do
        x[j] = A[{row, j}] 
    end 
end

-- Returns the dot product of two vectors of equal length.
task dot(is : ispace(int1d),
         x  : region(ispace(int1d), float),
         y  : region(ispace(int1d), float))
where
    reads(x, y)
do
    var sum : float = 0
    for i in is do
        sum += (y[i] * x[i])
    end
    return sum
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

-- Special version of saxpy where a linear combination of
-- ROWS r1 and r2 of a 2D region are stored in y.
task saxpy_2d(is    : ispace(int1d),
              r1    : uint64,
              r2    : uint64,
              x     : region(ispace(int2d), float),
              y     : region(ispace(int1d), float),
              a     : float)
where
    reads(x, y), writes(y)
do
    __demand(__vectorize)
    for i in is do 
        y[i] = x[{r1,i}] + a*x[{r2,i}]
        --c.printf("%f ", x[{r2,i}])  
    end
    --c.printf("\n")
end

-- Returns evaluation of linear function f(x) = w^T x + b.
-- Note: the bias term is included in vector w.
task f(is : ispace(int1d),
       w  : region(ispace(int1d), float),
       x  : region(ispace(int1d), float))
where 
    reads(x, w)
do
    var f_ : float = 0.0
    -- TODO(db) How do I vectorize this? 
    --__demand(__vectorize)
    for i in is do
        f_ += x[i] * w[i]
    end
    return f_
end

-- Gradient of hinge loss function.
-- is_c : index space for columns of A (dimension of input, n).
task hl_grad(is_c  : ispace(int1d),
             w     : region(ispace(int1d), float),
             x     : region(ispace(int1d), float), 
             y     : int32,
             lam   : float,
             eta   : float)
where 
    reads(x), 
    reads writes (w)
do
    var z = region(is_c, float)
    fill(z, 0.0)
    -- z += (-eta * lam) * w_t.
    saxpy(is_c, w, z, -eta * lam)
    if (y * f(is_c,w,x) < 1) then
        var z_ = region(is_c, float)
        fill(z_, 0.0)
        -- z_ += -eta * y_i * x_i
        saxpy(is_c, x, z_, -y * eta)
        -- z += z
        saxpy(is_c, z_, z, 1)
    end
    -- w += -z
    saxpy(is_c, z, w, -1)
end

-- Builds the kernel matrix. 
-- m : number of training examples. 
-- n : number of features. 
task compute_kernel(m      : uint64,
                    n      : uint64,
                    data   : region(ispace(int2d), float),
                    kernel : region(ispace(int2d), float))
where
    reads(data, kernel), writes(kernel)
do
    --var ker_is = ispace(int2d, {x = m, y = m}) 
    var r_is = ispace(int1d, m)
    var c_is = ispace(int1d, n)
    var y = region(c_is, float) -- Same dimension as features (n).   
    var sig : float = 64.0
    for i in r_is do
        for j in r_is do
            saxpy_2d(c_is, i, j, data, y, -1)
            var p : float = dot(c_is, y, y)
            kernel[{i,j}] = cmath.exp(p / (-2.0*sig))
            --c.printf("k: %f \n", kernel[{i,j}])
        end
    end
    do return end
end

-- Uses the dual formulation.
-- is : row index space corresponding to number of training examples.
-- num_rows : number of training examples.
-- num_features : number of features in each example.
task svm_train_dual(is           : ispace(int1d),
                    num_rows     : uint64,
                    num_features : int64,
                    data         : region(ispace(int2d), float),
                    labels       : region(ispace(int1d), float),
                    alpha        : region(ispace(int1d), float))
where 
    reads(data, labels, alpha), writes(alpha)
do
    var ker_ispace = ispace(int2d, {x = num_rows, y = num_rows}) 
    var K = region(ker_ispace, float) 
    -- Fill 2D region K.
    compute_kernel(num_rows, num_features, data, K)
    var grad = region(is, float)
    var k_ = region(is, float)
    var k_c = region(is, float)
    var lambda : float = 1.0 / (64.0 * num_rows)
    var num_outer_loops : uint32 = 40
    var scaling_factor : double = num_outer_loops * num_rows 
    var alpha_avg = region(is, float)
    fill(alpha_avg, 0.0)
    fill(alpha, 0.0)
    for ii = 1, num_outer_loops * num_rows do
        var ind : uint64 = cmath.floor((num_rows * (c.rand() / (c.RAND_MAX+0.1))))
        --c.printf("ind: %d\n", ind)
        regentlib.assert(ind <= num_rows, "invalid index")
        for ic in is do
            k_[ic] = K[{ind, ic}]
            k_c[ic] = K[{ic, ind}]
            grad[ic] = [float](num_rows * lambda * K[{ic, ind}] * alpha[ind])
        end
        var margin = labels[ind] * dot(is, k_, alpha) 
        if (margin < 1) then
            -- grad -= Y[i]*K[:,ind] 
            saxpy(is, k_c, grad, -labels[ind])
        end
        saxpy(is, grad, alpha, -1.0/sqrt(ii))
        saxpy(is, alpha, alpha_avg, 1.0) 
        --print(is, alpha)
    end
    fill(alpha, 0.0)
    saxpy(is, alpha_avg, alpha, (1.0 / scaling_factor))

    c.printf("Print alpha ... \n")
    print(is, alpha)
end

-- Trains a support vector machine.
-- c_is : column index space.
-- r_is : row index space.
-- data : training data.
-- labels : corresponding labels for training data instances. 
task svm_train(c_is     : ispace(int1d),
               r_is     : ispace(int1d),
               data     : region(ispace(int2d), float),
               labels   : region(ispace(int1d), float),
               weights  : region(ispace(int1d), float))
where 
    reads(data, labels),
    writes reads(weights)
do
    var eta : float = 0.05 --Step size. 
    var lam : float = 0.005 --Regularization. 
    -- Initialize weights vector.
    for cc in c_is do
        weights[cc] = [float](c.rand() / (c.RAND_MAX))
    end
    for i = 0, MAX_ITERS do
        c.printf("Iteration #: %d \n", i)
        var x = region(c_is, float)
        for ri in r_is do
            fill(x, 0.0)
            for ci in c_is do
                x[ci] = data[{ri, ci}]
            end 
            -- Uses the hinge loss to compute weight vector.
            -- Currently using stochastic gradient descent. 
            hl_grad(c_is, weights, x, labels[ri], lam, eta)
        end
    end
    print(c_is, weights)
end

-- Computes the accuracy of the kerenl SVM on the test dataset.
-- is_c : Index space corresponding to parameter dimension.
-- is_r : Index space corresponding to number of instances.  
-- w : Support vector parameters.
-- data : Sample data with n features.
-- lables : Correct classification of corresponding data instance.  
task svm_test_dual(c_is     : ispace(int1d),
                   p_train  : ispace(int1d),
                   p_test   : ispace(int1d),
                   alpha    : region(ispace(int1d), float),
                   data     : region(ispace(int2d), float),
                   labels   : region(ispace(int1d), float))
where 
    reads(data, labels, alpha)
do
    var sig : float = 64.0
    var acc : float = 0
    var total : uint64 = 0
    var x_ = region(c_is, float)
    var y_ : float = 0
    var k : float = 0
    -- Iterate over the entire test set to compute accuracy. 
    for ii in p_test do
        var class : double = 0.0
        total += 1
        for jj in p_train do
            saxpy_2d(c_is, ii, jj, data, x_, -1)
            var p : float = dot(c_is, x_, x_)
            k = cmath.exp(-p / (2*sig)) 
            --class += (alpha[jj] * labels[jj] * k)
            class += (alpha[jj] * k)
            --c.printf("%2.6f   %2.6f   %2.6f  --->  %2.6f \n", alpha[jj], labels[ii], k, class)
        end
        if (class*labels[ii] < 1) then
            c.printf("# %d is incorrect: %f -- %f. \n", ii, labels[ii], class)
        else 
            c.printf("# %d is correct: %f -- %f. \n", ii, labels[ii], class)
            acc += 1.0 
        end
    end 
    c.printf("# Correct: %f \n", acc)
    acc = [float](acc / total)
    c.printf("Accuracy: %f \n", acc)
    return acc
end

-- Computes the accuracy of the SVM on the test dataset.
-- is_c : Index space corresponding to parameter dimension.
-- is_r : Index space corresponding to number of instances.  
-- w : Support vector parameters.
-- data : Sample data with n features.
-- lables : Correct classification of corresponding data instance.  
task svm_test(is_c     : ispace(int1d),
              is_r     : ispace(int1d),
              w        : region(ispace(int1d), float),
              data     : region(ispace(int2d), float),
              labels   : region(ispace(int1d), float))
where 
    reads(data, labels, w)
do
    var acc : float = 0
    var total : uint64 = 0
    var x_ = region(is_c, float)
    var y_ : float = 0
    -- Iterate over the entire test set to compute accuracy. 
    for ir in is_r do
        fill(x_,0)
        c.printf("%d\n", ir)
        y_ =  labels[ir]
        for ic in x_.ispace do
            c.printf("%f ", data[{ir,ic}]) 
            x_[ic] = data[{ir,ic}]
        end
        total += 1
        if (y_ * f(w.ispace, w, x_) > 1) then
            c.printf("# %d is correct. \n", ir)
            acc += 1.0 
        else
            c.printf("# %d is incorrect: %f -- %f. \n", ir, y_, f(w.ispace,w,x_))
        end               
    end
    c.printf("# Correct: %f \n", acc)
    acc = [float](acc / total)
    c.printf("Accuracy: %f \n", acc)
    return acc
end

task toplevel()
    -- Load data from file.
    var FILE_NAME = "./data/toy_examples/ijcnn1.txt"
    var f_in = c.fopen(FILE_NAME, "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var nrows : uint64 = dim[0] -- Dimension of output vector.
    var ncols : uint64 = dim[1] -- Dimension of feature vector.
    
    var datafile : DataFile
    datafile.filename = FILE_NAME
    datafile.num_instances = nrows
    datafile.instance = [&data](c.malloc(nrows * [sizeof(data)]))
    datafile:parse()
    for i = 0, datafile.num_instances do
        c.printf("Label: %f \n", datafile.instance[i].label)
        for j = 0, datafile.instance[i].num_entries do
            c.printf("[%f] At index %d \n", 
                    datafile.instance[i].value[j], 
                    datafile.instance[i].indices[j])
        end
        c.printf("\n")
    end 
    var row_ispace = ispace(int1d, nrows)
    var col_ispace = ispace(int1d, ncols) -- Bias term included.
    c.printf("n: %d m: %d \n", nrows, ncols)

    -- Create region Matrix (A) and response vector (labels).   
    -- Note: Include bias term in A.
    var A = region(ispace(int2d, { x = nrows, y = ncols }), float)
    var labels = region(row_ispace, float)
    fill(A, 0.0)
    fill(labels, 0.0)

    var data : &float = [&float](c.malloc(ncols * [sizeof(float)]))
    for i = 0, nrows do
        A[{i, ncols-1}] = 1.0
        labels[i] = datafile.instance[i].label
        c.printf("Label: %f \n", labels[i])
        for j = 0, datafile.instance[i].num_entries do
            A[{i, datafile.instance[i].indices[j]}] = datafile.instance[i].value[j]
            c.printf("%f \n", A[{i, datafile.instance[i].indices[j]}])
        end
        c.printf("\n")
    end
    print(row_ispace, labels)
    c.fclose(f_in)

    var eta : float = 0.05
    var lam : float = 0.005

    -- TODO(db) Iterate over partition of A.
    var partitions = 2
    var data_ispace = ispace(int2d, { partitions, 1 })
    var label_ispace = ispace(int1d, partitions) 
    var data_partition = partition(equal, A, data_ispace) -- Test/train data partition
    var label_partition = partition(equal, labels, label_ispace) -- Test/train label partition. 
    var weights = region(col_ispace, float)


    var train = 0
    var test = 1 
    var alpha = region(label_partition[train].ispace, float)
    fill(alpha, 0.0)
    var m_training_examples = (nrows / partitions) 
    var n_features = ncols -- includes bias term.
    for block in data_ispace do
        if (block == [int2d]{train,0}) then
            --[[svm_train(col_ispace, label_partition[train].ispace, 
                      data_partition[block], label_partition[train], weights)--]]
            svm_train_dual(alpha.ispace, m_training_examples, n_features, 
                           data_partition[block], label_partition[train], alpha)
        elseif (block == [int2d]{test,0}) then
            c.printf("Testing! %f \n", labels[0])
            --[[svm_test(col_ispace, label_partition[1].ispace, weights, 
                     data_partition[block], label_partition[test])--]]
            svm_test_dual(col_ispace, label_partition[train].ispace,
                          label_partition[test].ispace, alpha, A, labels)
        end
    end
end

regentlib.start(toplevel)
