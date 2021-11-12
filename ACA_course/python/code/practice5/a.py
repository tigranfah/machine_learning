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

