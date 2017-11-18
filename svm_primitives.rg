import "regent"

-- Helper module to parse data stored in libsvm format. 
local DataFile = require("libsvm_format_parser")

-- C APIs
local c = regentlib.c
    
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

-- Computes the accuracy of the SVM on the test dataset.
-- is_c : Index space corresponding to parameter dimension.
-- is_r : Index space corresponding to number of instances.  
-- w : Support vector parameters.
-- data : Sample data with n features.
-- lables : Correct classification of corresponding data instance.  
task svm_test(is_c     : ispace(int1d),
              is_r     : ispace(int1d),
              w        : region(ispace(int1d), float),
              data     : region(ispace(int2d), float), -- TODO: change this to int2d
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
    for block in data_ispace do
        if (block == [int2d]{train,0}) then
            svm_train(col_ispace, label_partition[train].ispace, 
                      data_partition[block], label_partition[train], weights)
            --[[
            c.printf("Training!\n")
            for i = 0, MAX_ITERS do
                c.printf("ITER: %d \n", i)
                for r = 0, nrows do
                    -- TODO(db) Remove this hack to copy rows from A to x.
                    x[0] = 1.0
                    for i = 0, ncols-1 do
                        x[i+1] = A[{r, i}]
                    end
                    -- Use the hinge loss to compute weights vector.
                    -- Currently using stochastic gradient descent. 
                    hl_grad(col_ispace, weights, x, labels[r], lam, eta)
                end
            end
            print(col_ispace, weights)--]] 
        elseif (block == [int2d]{test,0}) then
            c.printf("Testing! %f \n", labels[0])
            svm_test(col_ispace, label_partition[1].ispace, weights, 
                     data_partition[block], label_partition[test])
        end
    end
end
--for d in data_partition[block] do
    --c.printf("Block: %d     index: (%d, %d)       value: %f \n", block, d.x, d.y, A[d])
--end

regentlib.start(toplevel)
