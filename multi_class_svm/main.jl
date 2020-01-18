include("mnistdata.jl")
include("SVM.jl")

"""
    Testing the model for a hard-coded λ
"""
# Load training data
X, Y = loadmnist(0:2,reduction=3,set=:train)

# Split intop training/dev set
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

# Generate Q-matrix
Q = Q_matrix(X_train, Y_train, lambda = 0.001)

# Generate kernel matrix
K = kernel_matrix(X_train)

# Solve dual problem
it = 1000
@time μ = train(Q, X_train, it = it)

# Predict on train data
y_hat = -1 * ones(length(Y_train))

for (i, xi) in enumerate(X_train)
    y_hat[i] = predict(K, μ, Y_train, X_train, xi)
end

mean(y_hat .== Y_train)

# Predict on test data
y_hat = -1 * ones(length(Y_test))

for (i, xi) in enumerate(X_test)
    y_hat[i] = predict(K, μ, Y_train, X_train, xi)
end

mean(y_hat .== Y_test)

"""
    Grid search to find best choice of regularization parameter
"""

lambda_grid = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1]
grid_search(lambda_grid)
