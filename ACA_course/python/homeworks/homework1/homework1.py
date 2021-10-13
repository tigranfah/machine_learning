def three_digit_sum(num):
	first_dig = num // 100
	second_dig = num % 100 // 10
	third_dig = num % 10
	#print(first_dig)
	#print(second_dig)
	#print(third_dig)
	return first_dig + second_dig + third_dig


def area_of_right_tri(a, b):
	return (a * b)/2


def arethmatic_prog(a1, a2, n):
	d = a2 - a1
	return a1 + (n - 1) * d


def century_from_year(year):
	return int(year + 100 - 1) // 100


def two_men(f, s):
	return s - 1, f - 1


def knights_moves(px, py):
	return (
		(px + 1, py + 2),
		(px + 2, py + 1),
		(px + 2, py - 1),
		(px + 1, py - 2),
		(px - 1, py - 2),
		(px - 2, py - 1),
		(px - 2, py + 1),
		(px - 1, py + 2)
	)

if __name__ == "__main__":
	print(three_digit_sum(567))
	print(area_of_right_tri(3, 4))
	print(arethmatic_prog(9, 4, 2))
	print(century_from_year(374))
	print(two_men(4, 7))
	print(knights_moves(6, 4))
