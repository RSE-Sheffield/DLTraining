import sys
import numpy as np
import caffe


if len(sys.argv) < 1:
    pritn("No path to image, use as follows: \n mnist_deply.py path/to/image.jpg")
    sys.exit()



caffe.set_mode_gpu()
caffe.set_device(0)

model_path = "code/practical_3/mnist_lenet_deploy.prototxt"
weights_path = "code/practical_2/mnist_lenet_iter_10000.caffemodel"
#Net loading parameters changed in Python 3
net = caffe.Net(model_path, 1, weights=weights_path)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

image = caffe.io.load_image(sys.argv[1], False) #Loads the image
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()
output_prob = output['loss'][0]  # the output probability vector for the first image in the batch

#Add the correct digits to correspond with the output probablility,
#in many casees you will have text classifications e.g. "dogs", "birds", etc.
digits_label = [0,1,2,3,4,5,6,7,8,9]

#Find the index with the highest probablility
highest_index = -1
highest_probability = 0.0

for i in range(0,9):
    if output_prob[i] > highest_probability:
        highest_index = i
        highest_probability = output_prob[i]

#Print our result
if highest_index < 0:
    print("Did not detect a number!")
else:
    print("Digit "+ str(digits_label[highest_index]) + " detected with " + str(highest_probability*100.0)+"%  probability.")
