# Load packages
using LinearAlgebra
using ProximalOperators
using Statistics
using Random
include("mnistdata.jl")

function scale_data(X_train)
    scl_vec = zeros(length(X_train) * length(X_train))
    idx = 1
    for (i, xi) in enumerate(X_train)
        for (j, xj) in enumerate(X_train)
                scl_vec[idx] = dot(xi, xj)
                idx += 1
        end
    end
    return X_train ./ mean(scl_vec)
end

function kernel_matrix(X)
    dim = length(X)
    K = zeros(dim, dim)
    for (i, xi) in enumerate(X)
        for (j, xj) in enumerate(X)
            K[i, j] = dot(xi, xj)^5
        end
    end
    return K
end

function kernel_operator(xi, xj)
    return dot(xi, xj)^5
end

function Q_matrix(X, Y; lambda = 1)
    n_classes = length(unique(Y))
    dim = length(X)
    Q = zeros(dim, dim)
    for i = 1:dim
        A_i = generate_A_i(n_classes, Y[i])
        for j = 1:dim
            A_j = generate_A_i(n_classes, Y[j])
            Q[i,j] = (A_i' * A_j) * kernel_operator(X[i], X[j])
        end
    end
    return (1/lambda) * Q
end

function generate_A_i(n_classes, class)
    a_i = zeros(n_classes)
    a_i[class + 1] = 1
    return a_i .- (1/n_classes) * ones(n_classes)
end

function train(Q, X; it = 10000)
    # Get dimension of X
    N = length(X)

    # Dimension of v
    N = length(X)

    # Initialize dual variable v
    μ = randn(N)

    # Number of iterations for CPGD
    n_iter = it * N

    # Initialize conjugate of hinge loss
    h_j = HingeLoss(ones(1), 1/N)
    h_jconj = Conjugate(h_j)

    # Coordinate gradient descent
    for i = 1:n_iter
        # Select coordinate j
        j = rand(1:length(μ))

        # Calculate γ
        γ = 1 / Q[j, j]

        # Calculate gradient
        gradμ_j = Q[:, j]' * μ

        # Perform gradient descent step w.r.t coordinate j
        μ_jgd = μ[j] - γ * gradμ_j

        # calculate new point v
        μ_j, _ = prox(h_jconj, [μ_jgd], γ)

        # Cache old point v
        μ_prev = μ

        # Update coordinate j
        μ[j] = μ_j[1]
    end
    return μ
end

function predict(K, μ, Y_train, X_train, x_test)
    # Get nbr of classes
    n_classes = length(unique(Y_train))

    # Initialize average confidence array
    c_λ = zeros(n_classes)

    # Calculate X_i^TXAμ given some feature map i.e. make use of Kernel operator
    # K(i, j)

    # Place holder for results
    k_vec = zeros(n_classes, length(X_train))

    for (i, xi) in enumerate(X_train)
        k_vec[:, i] = kernel_operator(xi, x_test) * generate_A_i(n_classes, Y_train[i])
    end

    # Calculate average confidence
    for class = 1:n_classes
        c_λ[class] = -generate_A_i(n_classes, class - 1)' * k_vec * μ
    end
    return argmax(c_λ) - 1
end

function train_test_split(X, Y; split = 0.25)
    # Shuffle data randomly
    idxs = randperm(length(X))
    X = X[idxs]
    Y = Y[idxs]

    split_idx = Int(round(length(X) * (1 - split)))
    X_train = X[1:split_idx]
    X_test = X[split_idx + 1:end]
    Y_train = Y[1:split_idx]
    Y_test = Y[split_idx + 1:end]

    return X_train, Y_train, X_test, Y_test
end

function grid_search(lambda_grid)
    # Extract a smaller part of training data
    X, Y = loadmnist(0:2,reduction=20,set=:train)

    # Split into train and dev set
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

    # Generate kernel matrix
    K = kernel_matrix(X_train)

    print("============= GRID SEARCH =============\n")

    for lambda in lambda_grid
        # Generate Q-matrix
        Q = Q_matrix(X_train, Y_train, lambda = lambda)

        # Solve dual problem
        it = 1000

        μ = train(Q, X_train, it = it)

        # Predict on train data
        y_hat = -1 * ones(length(Y_train))

        for (i, xi) in enumerate(X_train)
            y_hat[i] = predict(K, μ, Y_train, X_train, xi)
        end

        error_train = mean(y_hat .!= Y_train)

        # Predict on test data
        y_hat = -1 * ones(length(Y_test))

        for (i, xi) in enumerate(X_test)
            y_hat[i] = predict(K, μ, Y_train, X_train, xi)
        end

        error_test = mean(y_hat .!= Y_test)

        # Print outs
        print("Lambda: ", lambda, " Test error rate: ", error_test,
                        " Train error rate: ", error_train, "\n")

    end
end
