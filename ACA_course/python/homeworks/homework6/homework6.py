import os
import copy


class Matrix:

	def __init__(self, cols=None, rows=None, value=0, **kwargs):

		self._elements = []

		if cols and rows:
			self._elements = Matrix.list_from_dim(cols, rows, value)
		elif kwargs.get("array") != None and isinstance(kwargs["array"], list):
			self._elements = kwargs["array"]
		elif kwargs.get("file_path") != None and isinstance(kwargs["file_path"], str):
			self.__read_from_file(kwargs["file_path"])
		else:
			raise Exception("Please provide a valid argument.")

		self._cols = len(self._elements[0])
		self._rows = len(self._elements)

		self.__check_dim()

	def __check_dim(self):
		for i, row in enumerate(self._elements[1:]):
			if len(row) != self._cols:
				raise Exception(f"{i + 2} row has different length.")

	def __read_from_file(self, file_path):
		with open(file_path, "r") as file:
			lines = file.readlines()
			for line in lines:
				line = line.replace("\n", "")
				self._elements.append([int(i) for i in line.split(" ")])

	def _sub_array(self, x1, y1, x2, y2):
		if x2 - x1 <= 0 or y2 - y1 <= 0 or y1 < 0 or x1 < 0 or x2 < 0 or y2 < 0:
			raise Exception(f"Can get submatrix with {x1}, {y1}, {x2}, {y2}.")
		sub_array = []
		for i in range(x2 - x1):
			sub_array.append([])
			for j in range(y2 - y1):
				sub_array[-1].append(self[i + x1, j + y1])

		return sub_array

	def apply_inplace(self, func):
		for j in range(self._cols):
			for i in range(self._rows):
				self._elements[i][j] = func(self._elements[i][j])

	def apply(self, other, func):
		mat = Matrix(array=Matrix.list_from_dim(self._cols, self._rows))
		for j in range(self._cols):
			for i in range(self._rows):
				mat[j, i] = func(self[j, i], other[j, i])

		return mat

	def fill(self, dum):
		self.apply_inplace(lambda x : dum)

	def save_to_file(self, file_name):
		with open(file_name, "w") as file:
			for row in self._elements:
				file.write(" ".join([str(i) for i in row]))
				file.write("\n")

	def transpose(self):
		mat = Matrix(array=Matrix.list_from_dim(self._cols, self._rows))
		for i in range(self._cols):
			mat.elements[i] = [row[i] for row in self._elements]

		return mat

	def determinant(self):

		"""
			Is calculated using this formula.
			https://semath.info/src/determinant-five-by-five.html
		"""

		if self._cols != self._rows:
			raise Exception(f"The determinant cannot be calculated of the matrix with {self._cols} columns and {self._rows} rows.")
		if self._cols == 2 and self._rows == 2:
			return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
		else:
			det_sum = 0
			sign = 1

			multipliers = self._elements[0][:]
			for i, mul in enumerate(multipliers):
				elems = copy.deepcopy(self._elements[1:])
				for j in range(len(elems)):
					elems[j].pop(i)

				det_sum += sign * mul * Matrix(array=elems).determinant()
				sign *= -1

			return det_sum

	def trace(self):
		return sum([self[i, i] for i in range(3)])

	def submatrix(self, x1, y1, x2, y2):
		return Matrix(array=self._sub_array(x1, y1, x2, y2))

	def __add__(self, other):
		return self.apply(other, lambda x, y : x + y)

	def __mul__(self, other):
		return self.apply(other, lambda x, y : x * y)

	def __matmul__(self, other):
		mat = Matrix(array=Matrix.list_from_dim(self._rows, other.shape[1]))
		for j in range(mat.shape[1]):
			for i in range(mat.shape[0]):
				#print(sum(self._elements[j] + [row[i] for row in other.elements]))
				mat[j, i] = sum([x * y for x, y in zip(self._elements[j], [row[i] for row in other.elements])])

		return mat

	def __getitem__(self, pos):
		return self._elements[pos[0]][pos[1]]

	def __setitem__(self, pos, value):
		self._elements[pos[0]][pos[1]] = value

	def __str__(self):
		s = '---------MATRIX---------\n'
		s += '\n'.join(str(row) for row in self._elements)
		s += '\n'
		s += f'colums = {self.shape[0]}\nrows = {self.shape[1]}'
		s += '\n------------------------\n'
		return s
		#return "\n".join([" ".join(str(i) for i in row) for row in self._elements])

	def summation(self):
		return sum([sum(elems) for elems in self._elements])

	@property
	def shape(self):
		return self._rows, self._cols

	@property
	def elements(self):
		return self._elements

	@staticmethod
	def list_from_dim(rows, cols, value=0):
		return [[value for _ in range(cols)] for _ in range(rows)]


def save_to_file():
	m1 = Matrix(array=[[11, 9, 1], [24, 3, 8]])
	m1.save_to_file("mat1.txt")


def read_from_file():
	m1 = Matrix(file_path="mat1.txt")
	print(m1)


def matmul():
	m1 = Matrix(array=[[11, 9, 1], [24, 3, 8]])
	m2 = Matrix(array=[[11, 9], [24, 3], [31, 5]])
	# should be
	#	C1	C2
	#1	368	131
	#2	584	265
	print(m1 @ m2)


def transpose():
	m1 = Matrix(array=[[11, 9, 1], [24, 3, 8]])
	print(m1.transpose())


def submatrix():
	m1 = Matrix(array=[[11, 9, 1], [24, 3, 8]])
	print(m1.submatrix(0, 0, 2, 1))
	#print(m1._sub_array(0, 0, 2, 1))


def determinant():
	m1 = Matrix(array=[[8, 9, 1], [0, 3, 8], [4, 5, 6]])
	# the answer should be 100
	print(m1.determinant())

	m2 = Matrix(array=[[0, 4, 8, 9, 1],
					   [5, 4, 0, 3, 8],
					   [8, 3, 4, 5, 6],
					   [1, 3, 0, 3, 8],
					   [5, 4, 0, 3, 8]])

	# the answer should be 0
	print(m2.determinant())


def trace():
	m1 = Matrix(array=[[8, 9, 1], [0, 3, 8], [4, 5, 6]])
	print(m1.trace())


if __name__ == "__main__":
	#save_to_file()
	#read_from_file()
	#matmul()
	#transpose()
	#submatrix()
	#determinant()
	trace()
