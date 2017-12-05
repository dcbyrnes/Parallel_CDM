import "regent"

-- Helper module to parse data stored in libsvm format. 
local DataFile = require("libsvm_format_parser")

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
sqrt = regentlib.sqrt(double)
pow = regentlib.pow(double)
ceil = regentlib.ceil(double)
    
MAX_ITERS = 10

fspace Cluster {
    id          : uint64,
    centroid    : int1d, -- Centroid index. 
}

fspace Data {
    data      : float[256], -- Row of input matrix ...
    label     : float, -- Corresponding label ...          
    --kernel    : float[2000], -- Kernel wrt all other rows ...
    alpha     : float, -- Support Vector ...
    avg_alpha : float, -- Average Alpha ...
    centroid  : int1d, -- Designated centroid (processor ID) ...
    id        : uint64, -- Debugging purposes ...
}

terra read_index(f : &c.FILE, index : &uint64)
  return c.fscanf(f, "%llu\n", &index[0]) == 2
end

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
__demand(__inline)
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
__demand(__inline)
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
__demand(__inline)
task saxpy_2d(num_features : uint64,
              r1           : uint64,
              r2           : uint64,
              x1           : region(Data),
              x2           : region(Data),
              y            : region(ispace(int1d), float),
              a            : float)
where
    reads(x1.{data}, x2.{data}, y), writes(y)
do
    --__demand(__vectorize)
    for i = 0, num_features do 
        y[i] = x1[r1].data[i] + a * x2[r2].data[i]
        --c.printf("%f ", x1[r1].data[i])  
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
task compute_kernel(n            : uint64,
                    problem_data : region(Data),
                    kernel       : region(ispace(int2d), float))
where
    reads(problem_data), writes(kernel)
do
    var c_is = ispace(int1d, n)
    var y = region(c_is, float) -- Same dimension as features (n).   
    var sig : float = 0.5
    var idx : uint64 = 0
    for i in problem_data.ispace do
        for j in problem_data.ispace do
            saxpy_2d(n, i, j, problem_data, problem_data, y, -1)
            var p : float = dot(c_is, y, y)
            idx = j
            --problem_data[i].kernel[idx] = cmath.exp(p / (-2.0 * pow(sig,2)))
            kernel[{i,j}] = cmath.exp(p / (-2.0 * pow(sig, 2)))
            --c.printf("kernel: %f \n", problem_data[i].kernel[j])
            --c.printf("p: %f \n", p)
        end
    end
end

-- Uses the dual formulation.
-- is : row index space corresponding to number of training examples.
-- num_rows : number of training examples.
-- num_features : number of features in each example.
task svm_train_dual(valid_idx    : region(ispace(int1d), uint64), 
                    num_rows     : uint64,
                    num_features : int64,
                    problem_data : region(Data),
                    kernel       : region(ispace(int2d), float))
where 
    reads(problem_data.{id, label, alpha, avg_alpha}, valid_idx, kernel), 
    writes(problem_data.{alpha, avg_alpha})
do
    var lambda : float = 1.0 / (4 * num_rows)
    var num_outer_loops : uint32 = 40
    var scaling_factor : double = num_outer_loops * num_rows 
    var scale : float = 0
    --var boundary = clusters.bounds 
    --c.printf("%d ---> %d --> %d \n", boundary.lo, boundary.lo + 10, boundary.hi)

    for ii = 1, num_outer_loops * num_rows do
        var ind : uint64 = cmath.floor((num_rows * (c.rand() / (c.RAND_MAX+0.1))))
        ind = max(1, ind)
        regentlib.assert(ind <= num_rows, "invalid index")
        var random_index : int1d = ind 
        random_index = valid_idx[random_index]
        c.printf("random idx: %d \n",random_index)
        scale = num_rows * lambda * problem_data[random_index].alpha
        var margin : float = 0  
        var idx : uint64 = 0
        for inst in problem_data.ispace do
            --margin += problem_data[random_index].label * 
            --          problem_data[random_index].kernel[idx] * problem_data[inst].alpha
            margin += problem_data[random_index].label * 
                      kernel[{random_index, idx}] * problem_data[inst].alpha
            idx += 1 
        end
        --c.printf("margin: %f\n", margin) 
        if (margin < 1) then
            scale -= problem_data[random_index].label
        end
        scale *= 1 / [float](sqrt(ii))
        --c.printf("scale: %f \n", scale)
        idx = 0
        for inst in problem_data.ispace do
            --problem_data[inst].alpha += -scale * problem_data[random_index].kernel[idx]
            problem_data[inst].alpha += -scale * kernel[{idx, random_index}]
            --c.printf("W: %f \n", problem_data[inst].alpha)
            problem_data[inst].avg_alpha += problem_data[inst].alpha
            idx += 1
        end
    end

    for inst in problem_data.ispace do
        problem_data[inst].avg_alpha *= 1 / scaling_factor
    end
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
task svm_test_dual(num_features           : uint64,
                   num_training_instances : uint64,
                   num_test_cases         : uint64,
                   problem_data           : region(Data),
                   test_data              : region(Data))
where 
    reads(problem_data.{alpha, data}, test_data.{data, label})
do
    var column_ispace = ispace(int1d, num_features)
    --var test_ispace(int1d, num_test_cases)
    --var train_ispace(int1d, num_training_instances)
    var sig : float = 0.5
    var acc : float = 0
    var total : uint64 = 0
    var x_ = region(column_ispace, float)
    var y_ : float = 0
    var k : float = 0
    var class : double = 0.0
    -- Iterate over the entire test set to compute accuracy. 
    for ii in test_data.ispace do
        class = 0.0
        total += 1
        for jj in problem_data.ispace do
            saxpy_2d(num_features, ii, jj, test_data, problem_data, x_, -1)
            var p : float = dot(column_ispace, x_, x_)
            k = cmath.exp(-p / (2 * pow(sig,2))) 
            class += (problem_data[jj].alpha * k)
            --c.printf("%2.6f   %2.6f   %2.6f  --->  %2.6f \n", alpha[jj], labels[ii], k, class)
        end
        if (class * test_data[ii].label < 0) then
            c.printf("# %d is incorrect: %f -- %f. \n", ii, test_data[ii].label, class)
        else 
            c.printf("# %d is correct: %f -- %f. \n", ii, test_data[ii].label, class)
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

-- Loads a data file stored in LIBSVM format and returns 
task load_libsvm_format(file_name    : &int8,
                        problem_data : region(Data),
                        nrows        : uint64,
                        ncols        : uint64)
where
    reads writes(problem_data)
do
    var f_in = c.fopen(file_name, "rb")

    var datafile : DataFile
    datafile.filename = file_name
    datafile.num_instances = nrows
    datafile.instance = [&data](c.malloc(nrows * [sizeof(data)]))
    datafile:parse()
    -- Print data.
    for i = 0, datafile.num_instances do
        c.printf("Label: %f \n", datafile.instance[i].label)
        for j = 0, datafile.instance[i].num_entries do
            c.printf("[%f] At index %d \n", 
                    datafile.instance[i].value[j], 
                    datafile.instance[i].indices[j])
        end
        c.printf("\n")
    end 
    --[[
    for i = 0, nrows do
        data_matrix[{i, ncols-1}] = 1.0
        labels[i] = datafile.instance[i].label
        c.printf("%d.) Label: %f \n", i, labels[i])
        for j = 0, datafile.instance[i].num_entries do
            data_matrix[{i, datafile.instance[i].indices[j]}] = datafile.instance[i].value[j]
            --c.printf("%f \n", data_matrix[{i, datafile.instance[i].indices[j]}])
        end
        c.printf("\n")
    end
    --]]
    var ii : uint64 = 0
    for jj in problem_data.ispace do
        problem_data[jj].data[ncols-1] = 1.0
        problem_data[jj].label = datafile.instance[ii].label
        for j = 0, datafile.instance[ii].num_entries do
            problem_data[jj].data[datafile.instance[ii].indices[j]] = 
                                                                datafile.instance[ii].value[j]
        end 
        ii += 1
    end
    c.fclose(f_in)
end

task toplevel()
    -- Load TRAINING data from file.
    var training_datafile = "./data/toy_examples/ijcnn1.tr"
    var f_in = c.fopen(training_datafile, "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var nrows : uint64 = dim[0] -- Dimension of output vector.
    var ncols : uint64 = dim[1] -- Dimension of feature vector.
    var row_ispace = ispace(int1d, nrows)
    var col_ispace = ispace(int1d, ncols) -- Bias term included.
    c.printf("n: %d m: %d \n", nrows, ncols)
    c.fclose(f_in)
    var problem_data = region(ispace(ptr, nrows), Data) 

    -- Create region Matrix (train_data) and response vector (train_labels).   
    -- Note: Include bias term in train_data.
    load_libsvm_format(training_datafile, problem_data, nrows, ncols)

    -- Load TEST data from file.
    var testing_datafile = "./data/toy_examples/ijcnn1.t"
    f_in = c.fopen(testing_datafile, "rb")
    get_dimensions(f_in, dim)
    var ntest_rows : uint64 = dim[0] -- Dimension of output vector.
    var ntest_cols : uint64 = dim[1] -- Dimension of feature vector.
    var test_row_ispace = ispace(int1d, ntest_rows)
    var test_col_ispace = ispace(int1d, ntest_cols) -- Bias term included.
    c.printf("n: %d m: %d \n", ntest_rows, ntest_cols)
    c.fclose(f_in)
    var test_problem_data = region(ispace(ptr, ntest_rows), Data) 

    -- Create region Matrix (train_data) and response vector (test_labels).   
    -- Note: Include bias term in train_data.
    load_libsvm_format(testing_datafile, test_problem_data, ntest_rows, ntest_cols)

    var eta : float = 0.05
    var lam : float = 0.005

    var m_training_examples = nrows 
    var m_testing_examples = ntest_rows 
    var n_features = ncols -- includes bias term.

    var num_clusters = 8
    var cluster_mapping = "./src/clustering_results/kmeans_clustering_30_fake.tr"
    f_in = c.fopen(cluster_mapping, "rb")
    var index : uint64[1]
    var idx = 0
    for instance in problem_data do
        read_index(f_in, index)
        instance.id = idx
        instance.centroid = index[0]
        instance.alpha = 0
        instance.avg_alpha = 0
        idx += 1
    end
    c.fclose(f_in)

    var clusters = region(ispace(ptr, nrows), Cluster)
    f_in = c.fopen(cluster_mapping, "rb")
    idx = 0
    for clx in clusters do
        read_index(f_in, index)
        clx.centroid = index[0]
        clx.id = idx
        c.printf("clx: %d \n", clx.centroid)
        idx += 1
    end
    c.fclose(f_in)
    
    var colors = ispace(int1d, num_clusters)
    var data_partition = partition(problem_data.centroid, colors) 
    var cluster_partition = partition(clusters.centroid, colors)
    
    var partition_sizes = region(ispace(int1d, num_clusters), uint64)
    var ii_ = 0    
    for color in data_partition.colors do
        ii_ = 0
        for x in data_partition[color] do
            ii_ += 1
        end
        partition_sizes[color] = ii_
        c.printf("part size: %d \n", partition_sizes[color])
    end 
   
    var kernel_matrix = region(ispace(int2d, {x = nrows, y = nrows}), float) 
    -- Fill 2D region K.
    --compute_kernel(nrows, n_features, problem_data)
    for color in data_partition.colors do
        compute_kernel(n_features, data_partition[color], kernel_matrix)
        c.printf("%d go!\n", color)
    end

    --svm_train_dual(m_training_examples, n_features, problem_data)
    var valid_idx = region(ispace(int1d, nrows), uint64)
    var i_ = 0
    for color in data_partition.colors do
        i_ = 0
        fill(valid_idx, 0)
        for x in cluster_partition[color] do
            valid_idx[i_] = x.id
            c.printf("p: %d \n", valid_idx[i_])
            i_ += 1
        end
        svm_train_dual(valid_idx, partition_sizes[color], n_features, 
                        data_partition[color], kernel_matrix)
    end
    
    c.printf("Print alpha ... \n")
    for inst in problem_data do
        c.printf("alpha: %f \n", inst.avg_alpha)
    end

    --svm_test_dual(n_features, m_training_examples, m_testing_examples, 
    --              problem_data, test_problem_data)
    for color in data_partition.colors do
        svm_test_dual(n_features, m_training_examples, m_testing_examples, 
                    data_partition[color], test_problem_data)
    end
end

regentlib.start(toplevel)
