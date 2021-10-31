import math


def replace_char(string, i, value="1"):
	string = list(string)
	string[i] = value
	return "".join(string)


def beautiful_bin_str():
	string = input()

	steps = 0

	while True:
		#print("here")

		i = string.find("01010")

		if i != -1:
			string = replace_char(string, i+2)
			steps += 1
		else: break
	
	while True:
		j = string.find("010")

		if j != -1:
			string = replace_char(string, j)
			steps += 1
		else: break

	print(string)
	print(steps)


def string_power():

	s = input()
	k = int(input())

	if k == 0:
		return ""
	elif k > 0:
		for i in range(k-1):
			s += s
		return s
	elif k < 0:
		for i in range(1, len(s)+1):
			new_str = str()
			for j in range(-k):
				new_str += s[:i]
			if new_str == s:
				return s[:i]
		return "undefined"

if __name__ == "__main__":
	#beautiful_bin_str()
	print(string_power())
