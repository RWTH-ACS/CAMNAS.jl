module MatrixValidator

using LinearAlgebra

function m(matrix::Matrix)
    size(matrix, 1)
end

function n(matrix::Matrix)
    size(matrix, 2)
end

function is_quadratic(matrix::Matrix)
    m(matrix) == n(matrix)
end

function is_lu_decomposable(matrix::Matrix)
    try lu(matrix)
        true
    catch
        false
    end
end

function density(matrix::Matrix)
    non_zeroes = count(!iszero, matrix::Matrix)
    total_elements = size(matrix, 1) * size(matrix, 2)
    density = non_zeroes / total_elements
    return density
end

function condition(matrix::Matrix)
    cond(matrix)
end

end
