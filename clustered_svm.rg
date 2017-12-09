import "regent"

-- Helper module to parse data stored in libsvm format. 
local DataFile = require("libsvm_format_parser")

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
sqrt = regentlib.sqrt(double)
pow = regentlib.pow(double)
ceil = regentlib.ceil(double)
    
fspace Cluster {
    id          : uint64,
    centroid    : int1d, -- Centroid index. 
}

fspace Data {
    data      : float[256], -- Row of input matrix ...
    label     : float, -- Corresponding label ...          
    alpha     : float, -- Support Vector ...
    avg_alpha : float, -- Average Alpha ...
    centroid  : int1d, -- Designated centroid (processor ID) ...
    id        : uint64, -- Debugging purposes ...
}

---- Helper functions ----
-- Reads cluster assignments (indices) from file.
terra read_index(f : &c.FILE, index : &uint64)
  return c.fscanf(f, "%llu\n", &index[0]) == 2
end

-- Returns dimension of dataset, 
-- where number of rows in the first line and number
-- of columns is the second line.
terra get_dimensions(f : &c.FILE, d : &uint32)
    c.fscanf(f, "%d %d \n", &d[0], &d[1])
end

-- Reads data from file.
terra read_data(f : &c.FILE, n : uint64, input : &float)
    var x : float, y : float
    for  i = 0, n do
        c.fscanf(f, "%f", &input[i]) 
    end
    c.fscanf(f, "\n")
end

-- Prints vectors.
task print(is : ispace(int1d), 
           x : region(ispace(int1d), float))
    where 
        reads(x)
    do
    for i in is do
        c.printf("%f \n", x[i])
    end
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
    -- Print data --.
    for i = 0, datafile.num_instances do
        c.printf("Label: %f \n", datafile.instance[i].label)
        for j = 0, datafile.instance[i].num_entries do
            c.printf("[%f] At index %d \n", 
                    datafile.instance[i].value[j], 
                    datafile.instance[i].indices[j])
        end
        c.printf("\n")
    end 

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
----------------------------------------------------------

-- Returns the dot product of two vectors of equal length.
-- Assumes 'x' and 'y' have the same index space.
__demand(__inline)
task dot(x  : region(float),
         y  : region(float))
where
    reads(x, y)
do
    var sum : float = 0
    for i in x.ispace do
        sum += (y[i] * x[i])
    end
    return sum
end

-- DEPRECATED --
-- Sets y = a * x + y, where a is a scalar and x, y are vectors. 
--[[
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
--]]

-- SAXPY operation returns a linear combination of two vectors.
-- Requires that the vectors being combined are rows within the 
-- 'Data' region, specified by indices r1 and r2. Parameter 'a' 
-- scales data specified by row 'r2'.
__demand(__inline)
task saxpy_2d(num_features : uint64,
              alpha        : float,
              r1           : uint64,
              r2           : uint64,
              x1           : region(Data),
              x2           : region(Data),
              y            : region(float))
where
    reads(x1.{data}, x2.{data}), 
    reads writes(y)
do
    --__demand(__vectorize)
    var i : uint64 = 0
    for j in y.ispace do 
        y[j] = x1[r1].data[i] + alpha * x2[r2].data[i]
        i += 1
    end
end

-- Computes the kernel matrix. 
-- n : number of features. 
-- problem_data : Region of 'Data' that contains data instances.
-- kernel : 2D region that holds the kernel. 
task compute_kernel(n            : uint64,
                    problem_data : region(Data),
                    kernel       : region(ispace(int2d), float),
                    y            : region(float),
                    gamma        : float)
where
    reads(problem_data), 
    reads writes(y),
    writes(kernel)
do
    var scale : float = -1
    for i in problem_data.ispace do
        for j in problem_data.ispace do
            saxpy_2d(n, scale, i, j, problem_data, problem_data, y)
            var p : float = dot(y, y)
            kernel[{i,j}] = cmath.exp(p / (-2.0 * pow(gamma, 2)))
        end
    end
end

-- Trains the kernel svm using the s-pack algorithm.
-- num_rows : number of training examples.
-- num_features : number of features in each example.
task svm_train_dual(valid_idx    : uint64, 
                    num_rows     : uint64,
                    num_features : int64,
                    problem_data : region(Data),
                    kernel       : region(ispace(int2d), float))
where 
    reads(problem_data.{id, label, alpha, avg_alpha}, kernel), 
    writes(problem_data.{alpha, avg_alpha})
do
    var lambda : float = 1.0 / (4 * num_rows)
    var num_outer_loops : uint32 = 40
    var scaling_factor : double = num_outer_loops * num_rows 
    var scale : float = 0
    --var boundary = clusters.bounds 

    for ii = 1, num_outer_loops * num_rows do
        var ind : uint64 = cmath.floor((num_rows * (c.rand() / (c.RAND_MAX+0.1))))
        ind = max(1, ind)
        regentlib.assert(ind <= num_rows, "invalid index")
        var random_index : int1d = ind + valid_idx
        scale = num_rows * lambda * problem_data[random_index].alpha
        var margin : float = 0  
        var idx : uint64 = 0
        for inst in problem_data.ispace do
            margin += problem_data[random_index].label * 
                      kernel[{random_index, idx}] * problem_data[inst].alpha
            idx += 1 
        end
        if (margin < 1) then
            scale -= problem_data[random_index].label
        end
        scale *= 1 / [float](sqrt(ii))
        idx = 0
        for inst in problem_data.ispace do
            problem_data[inst].alpha += -scale * kernel[{idx, random_index}]
            problem_data[inst].avg_alpha += problem_data[inst].alpha
            idx += 1
        end
    end

    for inst in problem_data.ispace do
        problem_data[inst].avg_alpha *= 1 / scaling_factor
    end
end

-- Tests the SVM model on the test dataset and resturns the accuracy.
-- num_features : size of the features space.
-- num_training_instances : number of instances in the training dataset.
-- num_test_cases : number of instances in the test dataset. 
-- problem_data : region of type 'Data' containing training data information.
-- test_data : region of type 'Data' containing test problem data.
-- x : auxiliary region.
-- num_correct : region containing the number of test instances in each partition.
-- gamma : parameter in the RBF kernel.
task svm_test_dual(num_features           : uint64,
                   num_training_instances : uint64,
                   num_test_cases         : uint64,
                   problem_data           : region(Data),
                   test_data              : region(Data),
                   x                      : region(float),
                   num_correct            : region(ispace(int1d), uint64),
                   gamma                  : float)
where 
    reads(problem_data.{alpha, data}, test_data.{data, label}),
    reads writes(x),
    writes(num_correct)
do
    var total_correct : float = 0
    var acc : float = 0
    var total : uint64 = 0
    var k : float = 0
    var class : double = 0.0
    -- Iterate over the entire test set to compute accuracy. 
    for ii in test_data.ispace do
        class = 0.0
        total += 1
        for jj in problem_data.ispace do
            saxpy_2d(num_features, -1, ii, jj, test_data, problem_data, x)
            var p : float = dot(x, x)
            k = cmath.exp(-p / (2 * pow(gamma, 2))) 
            class += (problem_data[jj].alpha * k)
        end
        if (class * test_data[ii].label <= 0) then
            c.printf("# %d is incorrect: %f -- %f. \n", ii, test_data[ii].label, class)
        else 
            c.printf("# %d is correct: %f -- %f. \n", ii, test_data[ii].label, class)
            total_correct += 1.0 
        end
    end 
    c.printf("# Correct: %f \n", total_correct)
    if (total > 0) then
        acc = [float](total_correct / total)
    end
    c.printf("Accuracy: %f \n", acc)
    for x in num_correct do
        num_correct[x] = total_correct
    end
end

task toplevel()
    -- Load TRAINING data from file.
    var training_datafile = "./data/toy_examples/ijcnn1.tr"
    var f_in = c.fopen(training_datafile, "rb")
    var dim : uint32[2]
    get_dimensions(f_in, dim)
    var num_training_examples : uint64 = dim[0] -- Dimension of output vector.
    var num_features : uint64 = dim[1] -- Dimension of feature vector.
    var row_ispace = ispace(int1d, num_training_examples)
    var col_ispace = ispace(int1d, num_features) -- Bias term included.
    c.printf("n: %d m: %d \n", num_training_examples, num_features)
    c.fclose(f_in)

    var problem_data = region(ispace(ptr, num_training_examples), Data) 
    -- Create region Matrix (train_data) and response vector (train_labels).   
    -- Note: Include bias term in train_data.
    load_libsvm_format(training_datafile, problem_data, num_training_examples, num_features)

    -- Load TEST data from file.
    var testing_datafile = "./data/toy_examples/ijcnn1.t"
    f_in = c.fopen(testing_datafile, "rb")
    get_dimensions(f_in, dim)
    var num_testing_examples : uint64 = dim[0]
    var num_feat : uint64 = dim[1] -- Dimension of feature vector.
    var test_row_ispace = ispace(int1d, num_testing_examples)
    var test_col_ispace = ispace(int1d, num_feat) -- Bias term included.
    c.printf("n: %d m: %d \n", num_testing_examples, num_feat)
    c.fclose(f_in)

    var test_problem_data = region(ispace(ptr, num_testing_examples), Data) 
    -- Create region Matrix (train_data) and response vector (test_labels).   
    -- Note: Include bias term in train_data.
    load_libsvm_format(testing_datafile, test_problem_data, num_testing_examples, num_feat)

    var gamma : float = 0.5
    var num_clusters = 8
    var cluster_mapping = "./src/clustering_results/kmeans_clustering_30.tr"
    var testset_cluster_mapping = "./src/clustering_results/kmeans_clustering_30.t"
    
    -- Load cluster data --
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

    var clusters = region(ispace(ptr, num_training_examples), Cluster)
    f_in = c.fopen(cluster_mapping, "rb")
    idx = 0
    for clx in clusters do
        read_index(f_in, index)
        clx.centroid = index[0]
        clx.id = idx
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
        c.printf("part size: %d  <-- %d \n", partition_sizes[color], color)
    end 
   
    var kernel_matrix = region(ispace(int2d, 
            {x = num_training_examples, y = num_training_examples}), float)
    var kernel_colors = ispace(int2d, {x = num_clusters, y = 1}) 
    var kernel_partition = partition(equal, kernel_matrix, kernel_colors)
    var equal_data_partition = partition(equal, problem_data, colors)

    -- Auxiliary regions used during computations. 
    var auxiliary_region = region(ispace(ptr, num_clusters * num_features), float)
    var aux_part = partition(equal, auxiliary_region, colors)

    -- Compute Kernel Matrix --
    for color in kernel_partition.colors do
        compute_kernel(num_features, equal_data_partition[color.x], 
                        kernel_partition[color], aux_part[color.x], gamma)
        c.printf("%d go!\n", color)
    end

    -- Traing SVM -- 
    var valid_idx = region(ispace(int1d, num_training_examples), uint64)
    var i_ = 0
    var partition_size = [float](num_training_examples / num_clusters)
    var low = 0
    for color in data_partition.colors do
        --[[
        i_ = 0
        fill(valid_idx, 0)
        for x in cluster_partition[color] do
            valid_idx[i_] = x.id
            --c.printf("p: %d \n", valid_idx[i_])
            i_ += 1
        end 
        --]]
        low = color * partition_size
        svm_train_dual(low, partition_size, num_features, 
                    equal_data_partition[color], kernel_matrix)
    end
    
    -- Assign test instances to clusters --
    f_in = c.fopen(testset_cluster_mapping, "rb")
    idx = 0
    for instance in test_problem_data do
        read_index(f_in, index)
        instance.id = idx
        instance.centroid = index[0]
        idx += 1
    end
    c.fclose(f_in)
    var test_data_partition = partition(test_problem_data.centroid, colors) 
    
    var correct = region(ispace(int1d, num_clusters), uint64)
    var total_correct = partition(equal, correct, colors)

    -- Test SVM model -- 
    for color in data_partition.colors do
        svm_test_dual(num_features, num_training_examples, num_testing_examples,                                            data_partition[color], test_data_partition[color],
                           aux_part[color], total_correct[color], gamma)
    end
    var accuracy : float = 0
    for x in correct.ispace do
        accuracy += correct[x]
    end
    accuracy = [float](accuracy / num_testing_examples)
    c.printf("\n Complete Accuracy: %f \n", accuracy)
end

regentlib.start(toplevel)
