import pickle
original = open("./data/CONLL03/eng.train", "r")
output = open("./data/ner/eng.train.pre", "w")

block_start = True
for line in original:
	line = line.split()
	if (len(line) == 0):
		if block_start:
			block_start = False
			output.write("<eof> I None\n\n")

	if ( len(line) > 1 and line[0] != "-DOCSTART-"):
		if not block_start:
			block_start = True
			output.write('<s> O None')
		tmp = []
		tmp.append(line[0])
		tmp.append(line[3][0])
		tag = "None"
		if (len(line[3]) > 1):
			tag = line[3][2:]
		tmp.append(tag)
		output.write("%s\n" % " ".join(map(str, tmp)))
