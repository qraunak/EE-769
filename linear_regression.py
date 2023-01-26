
import numpy as np
import pandas as pd

def matrix_generator(sample_count, feature_dimension):
    matrix = np.random.randint(100, size=(sample_count, feature_dimension))
    return matrix


def column_generator(matrix, weight_vector, bais_w_0, noise_variance):
    noise=np.random.normal(loc=0, scale= noise_variance, size= noise_variance.shape)
#   weight_vector_transpose = weight_vector.transpose()
    xw = np.dot( matrix, weight_vector )
    return xw + bais_w_0 + noise


def calculate_estimate_y(matrix, weight_vector):
    return np.dot( matrix, weight_vector)


def mean_square_error(estimate_y, target_y):
    return (np.square(estimate_y - target_y)).mean(axis=0)


def estimate_weight(matrix, target_y , lambda_l2):
    transpose_matrix = matrix.transpose()
    matrix_multiplication = np.dot(transpose_matrix , matrix)
    identity_size = lambda_l2*np.identity(matrix_multiplication.shape[0])
    add_matrix = identity_size + matrix_multiplication
    add_matrix_inverse = np.linalg.inv(add_matrix)
    matrix_multiplication_target_y = np.dot(transpose_matrix, target_y)
    weight = np.dot(add_matrix_inverse , matrix_multiplication_target_y)
    pridicted_y = np.dot(transpose_matrix, weight)
    MSE= mean_square_error(pridicted_y, target_y)
    return weight, MSE, pridicted_y

def compute_l2_norm(weight_vector):
    return np.linalg.norm(weight_vector, ord=2)
    pass


def  compute_gradient_of_L2_norm(matrix, vector):
    pass

if __name__ == "__main__":
    ROW = 3
    COLUMN = 5

    matrix = matrix_generator(ROW,COLUMN)
    weight_vector = matrix_generator(COLUMN, 1)
    noise_variance = matrix_generator(ROW, 1)
    bais_w_0 = matrix_generator(ROW, 1)
    target_y = column_generator(matrix, weight_vector, bais_w_0, noise_variance)
    estimate_y=calculate_estimate_y(matrix, weight_vector)
    mse=mean_square_error(estimate_y, target_y)
    l2_norm=compute_l2_norm(weight_vector)
    weight, MSE, pridicted_y =estimate_weight(matrix,target_y,l2_norm)

