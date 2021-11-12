from collections import Counter


def good_pairs():
	nums = tuple(int(x) for x in input().split())
	nums_dict = {}
	good_pairs = []

	for i, n in enumerate(nums):
		if nums_dict.get(n) != None:
			good_pairs.extend([(ind, i) for ind in nums_dict[n]])
			nums_dict[n].append(i)
		else:
			nums_dict[n] = [i]

	print(good_pairs)
	return f"Number of good pairs is {len(good_pairs)}"


def good_pairs_only():
	nums = tuple(int(x) for x in input().split())
	nums_dict = {}
	good_pairs = 0

	for i, n in enumerate(nums):
		if nums_dict.get(n) != None:
			good_pairs += nums_dict[n]
			nums_dict[n] += 1
		else:
			nums_dict[n] = 1

	return f"Number of good pairs is {good_pairs}"


def unique_number_of_occ():
	nums = tuple(int(x) for x in input().split())
	dict_occ = {}
	for i in nums:
		if dict_occ.get(i) != None:
			dict_occ[i] += 1
		else: dict_occ[i] = 1

	return len(dict_occ) == len(set(dict_occ.values()))


def candies():
	candy_type = tuple(int(x) for x in input().split())
	unique_candies = set(candy_type)

	return min(len(unique_candies), len(candy_type)//2)


def anagrams():
	first_string = input()
	second_string = input()

	first_dict = {}
	second_dict = {}

	# what chars to remove from first string and second string
	remove_chars = ({}, {})

	# compute the unique occurances of string chars
	for char in first_string:
		if first_dict.get(char) != None:
			first_dict[char] += 1
		else: first_dict[char] = 1

	for char in second_string:
		if second_dict.get(char) != None:
			second_dict[char] += 1
		else: second_dict[char] = 1

	#print(first_dict)
	#print(second_dict)

	for k, v in first_dict.items():
		if second_dict.get(k) == None:
			remove_chars[0][k] = first_dict[k]
		else:
			if second_dict[k] < first_dict[k]:
				remove_chars[0][k] = first_dict[k] - second_dict[k]

	for k, v in second_dict.items():
		if first_dict.get(k) == None:
			remove_chars[1][k] = second_dict[k]
		else:
			if second_dict[k] > first_dict[k]:
				remove_chars[1][k] = second_dict[k] - first_dict[k]

	print(f"Chars to remove {remove_chars}")
	return sum(remove_chars[0].values()) + sum(remove_chars[1].values())


def find_words():

	print("input words") # such as - cat bt hat tree
	words = input().split()
	print("input chars") # such as - attach
	chars = input()

	char_dict = Counter(chars)

	total_chars = 0

	for word in words:
		word_dict = Counter(word)
		possible_to_make = True
		for k, v in word_dict.items():
			if char_dict.get(k) != None:
				if word_dict[k] > char_dict[k]:
					possible_to_make = False
			else: possible_to_make = False
			
		if possible_to_make:
			total_chars += len(word)

	return total_chars	


if __name__ == "__main__":
	#print(good_pairs())
	#print(good_pairs_only())
	#print(unique_number_of_occ())
	#print(candies())
	#print(anagrams())
	print(find_words())
