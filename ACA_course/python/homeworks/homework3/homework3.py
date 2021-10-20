import math


def is_prime(n):
	if not isinstance(n, int): return False
	if n < 1: return False
	for i in range(2, int(math.sqrt(n) + 1)):
		if n != i and n % i == 0:
			return False
	return True


def the_goldbach_conjection():
	num = int(input())
	assert num < 10000
	assert num % 2 == 0

	current_num = num - 1

	while True:
		if is_prime(current_num) and is_prime(num - current_num):
			break
		current_num -= 1

	return current_num, num - current_num


def is_palindrome(n):
	if n % 10 == n: # or n // 10 == 0:
		return True
	else:
		length = 10 ** (len(str(n)) - 1)
		first_dig = n // length
		last_digit = n % 10
		return is_palindrome((n - first_dig * length) // 10) if first_dig == last_digit else False


def palindrome_numbers():
	a, b = int(input()), int(input())
	return [n for n in range(a, b+1) if int(str(n)[::-1]) == n]
	# or, which is more expansive
	# return [n for n in range(a, b+1) if is_palindrome(n)]


def suffix_sums():
	# please give input in this form.
	# >>> 1 2 3 4 5 6
	inp = [float(i) for i in input().split()]
	return [sum(inp[i:]) for i in range(len(inp))]


def cyclic_shift():
	# please give input in this form.
	# >>> 1 2 3 4 5 6
	# >>> 2
	print("Input sequence.")
	seq = list(map(int, input().split()))
	#print("Its length")
	#N = int(input())
	print("How many times shift to right.")
	k = int(input())

	for i in range(k):
		seq.insert(0, seq.pop())

	return seq


def tree():
	n = int(input())
	assert n % 2 != 0

	for i in range(1, n + 1, 2):
		print(f"{' ' * ((n - i) // 2)}{'*' * i}")


if __name__ == "__main__":
	#print(the_goldbach_conjection())
	#print(palindrome_numbers())
	#print(suffix_sums())
	#print(cyclic_shift())
	tree()
