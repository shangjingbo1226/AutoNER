import pickle
original = open("./data/ner/eng.testb", "r")
output = open("./data/ner/eng.testb.pre", "w")
status = True
for line in original:
	line = line.split()
	if (len(line) == 0):
		if not status:
			output.write("<eof> I None\n")
			output.write("\n")
			status = True

	if ( len(line) > 1 and line[0] != "-DOCSTART-"):
		if status:
			output.write("<s> O None\n")
			status = False
		tmp = []
		tmp.append(line[0])
		chunk = line[3][0]
		if chunk == 'O':
			chunk = 'I'
		else:
			chunk = 'O'
		tmp.append(chunk)
		tag = "None"
		if (len(line[3]) > 1):
			tag = line[3][2:]
		tmp.append(tag)
		output.write("%s\n" % " ".join(map(str, tmp)))
