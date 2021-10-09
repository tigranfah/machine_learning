import math


def salary():
	a = int(input())
	b = int(input())
	c = int(input())

	maximum, minimum = a, a
	for i in [b, c]:
		if maximum < i:
			maximum = i
		if minimum > i:
			minimum = i

	return maximum - minimum
	# return max(a, b, c) - min(a, b, c)


def is_boring():
	num = int(input())
	return len(set(str(num))) == 1


def largest_number():
	num = int(input())
	digits = [int(i) for i in str(num)]
	return sorted(digits, reverse=True) > digits


def line_segment_intersection():
	a1 = float(input())
	b1 = float(input())
	a2 = float(input())
	b2 = float(input())
	l1 = abs(a1 - b1) # lengths
	l2 = abs(a2 - b2)
	s1 = min(a1, b1) # start points
	s2 = min(a2, b2)
	print(s1, l1, s2, l2)
	#s1 + l1 - s2
	length1 = b1 - b2 if b1 - b2 > 0 else 0
	length2 = b2 - b1 if b2 - b1 > 0 else 0
	cond = s1 + l1 - s2 - length1 if s1 < s2 else s2 + l2 - s1 - length2
	return cond if cond > 0 else 0


def number_of_div():
	num = int(input())
	num_of_div = 2
	for i in range(2, num // 2 + 1):
		if num % i == 0: num_of_div += 1
	return num_of_div


def quadratic_eq():
	a = float(input())
	b = float(input())
	c = float(input())
	if a == 0 and b == 0:
		if c == 0: print("Infinite solutions")
		else: print("No solutions")
		return

	D = b ** 2 - 4 * a * c
	print(f"Discriminant is {D}")
	if a == 0:
		print(f"Non-quadratic equation.")
		x = None
		if a == 0:
			x = -c/b
			print(f"One solution x : {x}")
	else:
		if D < 0:
			print("No solutions.")
		else:
			x1 = (-b + math.sqrt(D)) / 2 * a
			x2 = (-b - math.sqrt(D)) / 2 * a
			print(f"Two solutions x1 : {x1}, x2 : {x2}")


if __name__ == "__main__":
	#print(salary())
	#print(is_boring())
	#print(largest_number())
	print(line_segment_intersection())
	#print(number_of_div())
	#quadratic_eq()
