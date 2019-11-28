# Create YFCC-GEO100 splits (5% val, 20% test, 75% train)
import random

dataset_root = '/media/ssd2/YFCC100M-GEO100/'
val_file = open(dataset_root + 'splits/val.txt', 'w')
test_file = open(dataset_root + 'splits/test.txt', 'w')
train_file = open(dataset_root + 'splits/train.txt', 'w')


img_names = []

for line in open(dataset_root + 'photo2gps.txt', 'r'):
	img_name = line.split(' ')[0]
	img_names.append(img_name)

random.shuffle(img_names)
print(len(img_names))

num_val = int(len(img_names) * 0.05)
num_test = int(len(img_names) * 0.2)
train_img = 0

for i,img in enumerate(img_names):
	if i < num_val:
		val_file.write(img + '\n')
	elif i < num_val + num_test:
		test_file.write(img + '\n')
	else:
		train_file.write(img + '\n')
		train_img+=1


print("Num train: " + str(train_img) + ' test: ' + str(num_test) + ' val: ' + str(num_val))
