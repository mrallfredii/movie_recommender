# start module Alfred_Project
module Alfred_Project

# Imports
using DataFrames
using CSV
using Plots
using Statistics
using Random
using Flux

# Read the dataset from a CSV file
function read_csv()
    # Check file
    #isfile("./csv/movies.csv")

    # Load the movies.csv file into a DataFrame
    movies_df = CSV.read("./csv/movies.csv", DataFrame)

    # Save genres of movies
    y =  movies_df.genres

    # Load the rating.csv file into a DataFrame
    rating_df = CSV.read("./csv/ratings.csv", DataFrame)

    # Get the fisrt 3 columns to Matrix
    rating = Matrix(rating_df[:, 2:3])

    # Load the tags.csv file into a DataFrame
    tags_df = CSV.read("./csv/tags.csv", DataFrame)

    # Get the fisrt 3 columns to Matrix
    #tags = Matrix(tags_df[:, 1:2])

    # Combine the to Matrix to a global one x
    #X = [rating; tags]
    X = rating

    # Resize to dimension of movies
    X = X[1:length(y), :]
    
    # Display the first 5 rows of the DataFrame
    #println(X)

    return X, y
end

function split(X, y::AbstractVector; dims=1, ratio_train=0.8, kwargs...)
    n = length(y)
    size(X, dims) == n || throw(DimensionMismatch("Error"))

    n_train = round(Int, ratio_train*n)
    i_rand = randperm(n)
    i_train = i_rand[1:n_train]
    i_test = i_rand[n_train+1:end]

    return selectdim(X, dims, i_train), y[i_train], selectdim(X, dims, i_test), y[i_test]
end

function normalize(X_train, X_test; dims=1, kwargs...)
    col_mean = mean(X_train; dims)
    col_std = std(X_train; dims)

    return (X_train .- col_mean) ./ col_std, (X_test .- col_mean) ./ col_std
end

# Function binary matrix for knowing the data
function onehot(y, classes)
    y_onehot = falses(length(classes), length(y))
    for (i, class) in enumerate(classes)
        y_onehot[i, y .== class] .= 1
    end
    return y_onehot
end

#  Finds the index of its maximum value
onecold(y, classes) = [classes[argmax(y_col)] for y_col in eachcol(y)]

# Prepare the data
function prepare_data(X, y; do_normal=true, do_onehot=true, kwargs...)
    X_train, y_train, X_test, y_test = split(X, y; kwargs...)

    if do_normal
        X_train, X_test = normalize(X_train, X_test; kwargs...)
    end

    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_test = onehot(y_test, classes)
    end

    return X_train, y_train, X_test, y_test, classes
end

#  Neural network
function network_model(trainX, trainy, testX, testy)
   
    # Create the model
    m = Chain(Dense(trainX, 128, relu), Dense(128, 32, relu), Dense(32, 1))

    # compile the model
    # Cross-entropy is a common loss function used in classification problems, where the goal is to predict a discrete class.
    loss(x, y) = Flux.mse(m(x), y)

    # ADAM optimizer
    optimizer = ADAM()

    # Train the model
    epochs = 20
    train_loss = []
    test_loss = []
    for i in 1:epochs
        train_loss_ = []
        Flux.train!(loss, params(m), train, optimizer)
        
        for (x, y) in train
            train_loss_ = [train_loss_; loss(x, y)]
        end

        test_loss_ = []
        
        for (x, y) in test
            test_loss_ = [test_loss_; loss(x, y)]
        end
        
        push!(train_loss, mean(train_loss_))
        push!(test_loss, mean(test_loss_))
    end

    # Evaluate the model
    train_loss = Flux.evaluate(loss, zip(train_input, train_target))
    test_loss = Flux.evaluate(loss, zip(test_input, test_target))
    println("Train loss: ", train_loss)
    println("Test loss: ", test_loss)

end

# MAIN - test

# Read CSV dataframe
X, y = read_csv()
#println(first(df, 5))
#display(df)

# 
X_train, y_train, X_test, y_test, classes = prepare_data(X', y; dims=2)

# ALS
data = network_model(X_train, y_train, X_test, y_test)

####################################################################
# Train and test values files
#X_train, y_train, X_test, y_test = split(X, y)
#print(X_test)

# Normalized datasets
#X_train, X_test = normalize(X_train, X_test)
#print(X_test)

# One hot encoder
#classes = unique(y)
#y_one = onehot(y, classes)
#print(y_one)

#isequal(onecold(onehot(y, classes), classes), y)

######################################################################

# end module Alfred_Project
end 
