# Create ordered vocab for YFCC-GEO100

dataset_root = '/media/ssd2/YFCC100M-GEO100/'
ordered_vocab_file = open(dataset_root + 'ordered_vocab.txt','w')


tags = []

for line in open(dataset_root + 'photo2gps.txt', 'r'):
	img_name = line.split(' ')[0]
	tag = img_name.split('/')[0]
	if tag not in tags:
		tags.append(tag)

for tag in tags:
	ordered_vocab_file.write(tag + '\n')