from collections import Counter


def find_disappeared_numbers():
	# in range [1, n] such as 4,3,2,7,8,2,3,1
	nums = tuple(int(x) for x in input().split(","))
	appeared_dict = Counter(nums)
	print(appeared_dict)
	print(sum(appeared_dict.values()))
	disappeared = tuple(n for n in range(1, sum(appeared_dict.values())+1) 
						if appeared_dict.get(n) == None)
	return disappeared


def is_typed(row, word_dict):
	for key in word_dict.keys():
		if row.get(key) == None:
			return False
	return True


def keyboard_row(words):
	first_row = Counter("qwertyuiop")
	second_row = Counter("asdfghjkl")
	third_row = Counter("zxcvbnm")

	can_be_typed = []

	for w in words:
		word_dict = Counter(w.lower())
		for row in (first_row, second_row, third_row):
			if is_typed(row, word_dict):
				can_be_typed.append(w)
				break

	return can_be_typed


# printing matrix
def pprint(mat):
	for row in mat:
		print(row)


def transpose(mat):
	transposed = []
	i = 0
	for i in range(len(mat[i])):
		transposed.append([])
		for j in range(len(mat)):
			transposed[i].append(mat[j][i])

	return transposed


def reshape(mat, r, c):
	# expected shape length
	ex_length = len(mat[0]) * len(mat)
	if ex_length != r * c:
		if len(mat) == c:
			r = ex_length // c
		elif len(mat[0]) == r:
			c = ex_length // r
		else:
			raise Exception(f"{mat} can't be reshaped to ({c}, {r}) shape.")

	reshaped = []
	flatten_mat = iter(a for row in mat for a in row)
	for i in range(r):
		reshaped.append([])
		for j in range(c):
			reshaped[i].append(next(flatten_mat))
	
	return reshaped


class Board:

	# assume layout has rows of equal length.
	def __init__(self, layout):
		self._layout = layout
		self._cols = len(layout)
		self._rows = len(layout[0])
		self._ship_positions = []

	def find_ship(self, found_pos, x, y):
		unit_left = self.get(x+1, y)
		unit_down = self.get(x, y+1)
		if unit_left and unit_left == "X":
			found_pos.append((x+1, y))
			self.find_ship(found_pos, x+1, y)
		elif unit_down and unit_down == "X":
			found_pos.append((x, y+1))
			self.find_ship(found_pos, x, y+1)

		return

	def get_ship_count(self):
		ship_count = 0
		for j in range(self._cols):
			for i in range(self._rows):
				if (i, j) in self._ship_positions: continue
				curr_unit = self.get(i, j)
				if curr_unit and curr_unit == "X":
					ship_pos = [(i, j)]
					self.find_ship(ship_pos, i, j)
					self._ship_positions.extend(ship_pos)
					ship_count += 1
		return ship_count

	def get(self, x, y):
		if x >= 0 and x < self._rows and y >= 0 and y < self._cols:
			return self._layout[y][x]
		return None


if __name__ == "__main__":
	#print(find_disappeared_numbers())
	#print(keyboard_row(["Hello", "Alaska", "Dad", "Peace"]))
	#print(keyboard_row(["adsdf","sfd"]))
	#pprint(transpose([[1,2,3], [4,5,6], [7,8,9]]))
	#pprint(reshape([[1,2], [3,4]], 1, 4))
	#pprint(reshape([[1,2], [3,4]], 2, 4))

	lay = [["X",".",".","X"],
		   [".",".",".","X"],
		   [".",".",".","X"]]

	lay = [["X",".","X",".","X"],
		   [".",".","X",".","X"],
		   ["X",".",".",".","X"]]

	board = Board(lay)
	print(board.get_ship_count())
