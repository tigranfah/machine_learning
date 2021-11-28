import sys
import os

sys.path.insert(0, os.path.join("../homework6"))

from homework6 import Matrix


def convolution():

	img = Matrix(array=[[0, 1, 1, 1, 0, 0, 0],
						[0, 0, 1, 1, 1, 0, 0],
						[0, 0, 0, 1, 1, 1, 0],
						[0, 0, 0, 1, 1, 0, 0],
						[0, 0, 1, 1, 0, 0, 0],
						[0, 1, 1, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0]])

	kernel = Matrix(array=[[1, 0, 1],
							[0, 1, 0],
							[1, 0, 1]])

	new_array = []

						# cols
	for i in range(img.shape[1] - kernel.shape[1] + 1):
						# rows
		new_array.append([])
		for j in range(img.shape[0] - kernel.shape[0] + 1):
			submat = img.submatrix(i, j, i + kernel.shape[1], j + kernel.shape[0])
			mul = submat * kernel
			new_array[-1].append(mul.summation())

	print(Matrix(array=new_array))


def rotate():

	mat = Matrix(array=[[0, 1, 1, 1, 0, 0, 0],
						[0, 0, 1, 1, 1, 0, 0],
						[0, 0, 0, 1, 1, 1, 0],
						[0, 0, 0, 1, 1, 0, 0],
						[0, 0, 1, 1, 0, 0, 0],
						[0, 1, 1, 0, 0, 0, 0],
						[1, 1, 0, 0, 0, 0, 0]])

	array = [mat.elements[i][-(i+1):] for i in range(len(mat.elements))]

	for i in range(len(mat.elements)):
		row_to_set = mat.elements[i][:-(i+1)] + array[i]
		for j in range(len(mat.elements[i])):
			mat.elements[j][-(i+1)] = row_to_set[j]

	print(mat)


if __name__ == "__main__":
	#convolution()
	rotate()
