import os


class Matrix:

	def __init__(self, cols=None, rows=None, value=0, **kwargs):

		self._elements = []

		if cols and rows:
			self._elements = Matrix.list_from_dim(cols, rows, value)
		elif kwargs.get("array") != None and isinstance(kwargs["array"], list):
			self._elements = kwargs["array"]
		elif kwargs.get("file_path") != None and isinstance(kwargs["file_path"], str):
			self.__read_from_file(file_path)
		else:
			raise Exception("Please provide a valid argument.")

		self._cols = len(self._elements)
		self._rows = len(self._elements[0])

		self.__check_dim()

	def __check_dim(self):
		for i, row in enumerate(self._elements[1:]):
			if len(row) != self._rows:
				raise Exception(f"{i + 2} row has different length.")	

	def __read_from_file(self, file_path):
		with open(file_path, "r") as file:
			lines = file.readlines()
			for line in lines:
				line = line.replace("\n", "")
				self._elements.append([int(i) for i in line.split(" ")])

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


	def transpose(self, other):
		pass

	def __add__(self, other):
		return self.apply(other, lambda x, y : x + y)

	def __mul__(self, other):
		return self.apply(other, lambda x, y : x * y)

	def __getitem__(self, pos):
		return self._elements[pos[0]][pos[1]]

	def __setitem__(self, pos, value):
		self._elements[pos[0]][pos[1]] = value

	def __str__(self):
		return "\n".join([" ".join(str(i) for i in row) for row in self._elements])
	
	@staticmethod
	def list_from_dim(cols, rows, value=0):
		return [[value for _ in range(rows)] for _ in range(cols)]
		

#m = Matrix([[11, 9, 1], [24, 3, 8], [31, 5, 0]])
#m.save_to_file("mat.txt")

#m = Matrix(file_path="mat.txt")

m1 = Matrix(3, 4, value=1)
m2 = Matrix(3, 4, value=2)
#print(m1)
#print(m2)
print(m1 * m2)
