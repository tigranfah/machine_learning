import math


def digit_product():
	num = int(input())
	prod = 1
	while True:
		if num % 10 != 0:
			prod *= num % 10
		if num % 10 == num:
			break
		num = num // 10
	return prod


def largest_power_of_three():
	num = int(input())
	powers_of_three = 1
	while num >= powers_of_three * 3:
		powers_of_three *= 3
	return powers_of_three


def triangle():
	a, b, c = int(input()), int(input()), int(input())
	if a >= b + c or b >= a + c or c >= a + b:
		return "No Triangle"

	big_side = max(a, b, c)
	small_side = min(a, b, c)
	middle_side = (a+b+c) - big_side - small_side

	if big_side**2 > small_side**2 + middle_side**2:
		return "Obtuse Triangle"
	elif big_side**2 < small_side**2 + middle_side**2:
		return "Acute Triangle"
	
	return "Right Triangle"	


# this function is for the exercise the_root_of_the_number
def sum_of_digits(num):
	s = 0
	while True:
		if num % 10 != 0:
			s += num % 10
		if num % 10 == num:
			break
		num = num // 10
	return s

def the_root_of_the_number():
	num = int(input())
	while True:
		s_of_digs = sum_of_digits(num)
		print(s_of_digs)
		if s_of_digs < 10: break
		num = s_of_digs

def number_of_div():
	num = int(input())
	if num == 1: return 1

	# a positive int x > 1 is at least div by 1 and itself
	num_of_div = 2

	# current divisor
	cur_div = num // 2
	while cur_div != 1:
		if num % cur_div == 0: num_of_div += 1
		cur_div -= 1

	return num_of_div


def quadratic_eq():
	a = float(input())
	b = float(input())
	c = float(input())

	if a == 0:
		print(f"Non-quadratic equation.")
		if b == 0:
			if c == 0:
				print("Infinite solutions")
				return
			print("No solutions")
			return		
		x = None
		if a == 0:
			x = -c/b
			print(f"One solution x : {x}")
	else:
		print(f"Quadratic equation.")
		D = b ** 2 - 4 * a * c
		print(f"Discriminant is {D}")
		if D < 0:
			print("No solutions.")
		else:
			x1 = (-b + math.sqrt(D)) / 2 * a
			x2 = (-b - math.sqrt(D)) / 2 * a
			print(f"Two solutions x1 : {x1}, x2 : {x2}")


if __name__ == "__main__":
	#print(digit_product())
	#print(largest_power_of_three())
	#print(triangle())
	#the_root_of_the_number()
	#print(number_of_div())
	quadratic_eq()
