import pickle
original = open("./data/CONLL03/eng.testb", "r")
output = open("./data/ner/eng.testb.pre", "w")

for line in original:
	line = line.split()
	if (len(line) == 0):
		output.write("\n")

	if ( len(line) > 1 and line[0] != "-DOCSTART-"):
		tmp = []
		tmp.append(line[0])
		tmp.append(line[3][0])
		tag = "None"
		if (len(line[3]) > 1):
			tag = line[3][2:]
		tmp.append(tag)
		output.write("%s\n" % " ".join(map(str, tmp)))
