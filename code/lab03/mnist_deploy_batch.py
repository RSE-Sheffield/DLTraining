import sys
import numpy as np
import caffe


if len(sys.argv) < 1:
    pritn("No path to image, use as follows: \n mnist_deply.py path/to/image.jpg [more path to images ...]")
    sys.exit()

#User passed in this many images
num_images = len(sys.argv) - 1
#Get an array of image names
image_paths = sys.argv[1:]



model_path = "code/lab03/mnist_lenet_deploy.prototxt"
weights_path = "code/lab02/mnist_lenet_iter_10000.caffemodel"
#Net loading parameters changed in Python 3
net = caffe.Net(model_path, 1, weights=weights_path)

#Reshape the data blob to support batching
net.blobs['data'].reshape(len(image_paths), 1, 28,28)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

for index, path in enumerate(image_paths):
    image = caffe.io.load_image(path, False) #Loads the image
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[index, ...] = transformed_image

# Turn on GPU mode and use device zero for training
caffe.set_mode_gpu()
caffe.set_device(0)

### perform classification
output = net.forward()

#Specify labels to correspond with the output probablility,
#in many casees you will have text classifications e.g. "dogs", "birds", etc.
digits_label = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

#Get output probability vectors
for index, output_prob in enumerate(output['loss']) :

    #Find the index with the highest probablility
    highest_index = -1
    highest_probability = 0.0

    for i in range(10):
        if output_prob[i] > highest_probability:
            highest_index = i
            highest_probability = output_prob[i]

    #Print our result
    if highest_index < 0:
        print("Sample {:d} {:s}: Did not detect a number!".format(index, image_paths[index]) )
    else:
        print("Sample {:d}, {:s}: Digit {:s} detected with {:.2f}%  probability.".format( index, image_paths[index], digits_label[highest_index], highest_probability*100.0) )
