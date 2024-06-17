class Matrix:
    def __init__(self, matrix: list):
        self.matrix = matrix
        if matrix is not None:
            self.num_rows = len(matrix)
            self.num_cols = len(matrix[0])
        else:
            self.num_rows = 0
            self.num_cols = 0

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.matrix])

    def identity_matrix(self, size):
        identity_mat = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        return Matrix(identity_mat)

    def scale(self, scalar):
        scaled_mat = [[self.matrix[i][j] * scalar for j in range(self.num_cols)] for i in range(self.num_rows)]
        return Matrix(scaled_mat)
    
    def transpose(self):
        transposed_mat = [[self.matrix[j][i] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return Matrix(transposed_mat)

    def add(self, other_matrix):
        if self.num_rows != other_matrix.num_rows or self.num_cols != other_matrix.num_cols:
            raise ValueError("Attempting to add two matrices with different size!")
        added_mat = [[self.matrix[i][j] + other_matrix.matrix[i][j] for j in range(self.num_cols)] for i in range(self.num_rows)]
        return Matrix(added_mat)
    
    def dot(self, other_matrix):
        if self.num_cols != other_matrix.num_rows:
            raise ValueError("Attempting to dot multiply two matrices which are not aligned!")
        dotted_mat = [[sum(self.matrix[i][k] * other_matrix.matrix[k][j] for k in range(self.num_cols)) for j in range(other_matrix.num_cols)] for i in range(self.num_rows)]
        return Matrix(dotted_mat)

    def det(self):
        if self.num_cols != self.num_rows:
            raise ValueError("Attempting to compute determinant of a non-squared matrix!")
        if self.num_rows == 1:
            return self.matrix[0][0]
        if self.num_rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        
        determinant = 0
        for col in range(self.num_cols):
            submatrix = self._get_submatrix_(0, col)
            determinant += ((-1) ** col) * self.matrix[0][col] * submatrix.det()
        
        return determinant
    
    def _get_submatrix_(self, row, col):
        submatrix = [self.matrix[i][:col] + self.matrix[i][col+1:] for i in range(self.rows) if i != row]
        return Matrix(submatrix)
    
    def inverse(self):
        if self.num_cols != self.num_rows:
            raise ValueError("Attempting to invert a non-squared matrix!")
        n = self.num_rows
        identity_matrix = self.identity_matrix(n)
        augmented_mat = [self.matrix[i] + identity_matrix[i] for i in range(n)]

        for i in range(n):
            if augmented_mat[i][i] == 0:
                for j in range(i + 1, n):
                    if augmented_mat[j][i] != 0:
                        augmented_mat[i], augmented_mat[j] = augmented_mat[j], augmented_mat[i]
                        break
                else:
                    raise ValueError("Attempting to invert a singular matrix!")

            pivot = augmented_mat[i][i]
            for j in range(2 * n):
                augmented_mat[i][j] /= pivot

            for j in range(n):
                if i != j:
                    factor = augmented_mat[j][i]
                    for k in range(2 * n):
                        augmented_mat[j][k] -= factor * augmented_mat[i][k]

        inversed_mat = [row[n:] for row in augmented_mat]
        return Matrix(inversed_mat)
