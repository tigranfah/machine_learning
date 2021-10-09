def sum_of_digits(n): # n is a three digit number
	return sum([int(a) for a in str(n)])

print(sum_of_digits(123))
