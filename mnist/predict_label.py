# coding: utf-8
import caffe
import scipy.io as sio

caffe.set_mode_cpu()
root = 'F:\mnist'
model_def = root + '\mnist_deploy.prototxt'
weights_def = root + '\mnist__iter_10000.caffemodel'
net = caffe.Net(model_def, weights_def, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)

label = sio.loadmat(root + '/test_data/test_label.mat')
label_o = label['labels'][0]
print label_o

pred = []
count = 0
for i in range(10000):
    im = caffe.io.load_image(root + '/test_data/' + str(i) + '.jpg', color = False)
    # print im.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    pred_x = net.blobs['prob'].data[0].flatten().argmax()
    pred.append(pred_x)
    if pred_x == label_o[i]:
        count += 1

print 'prediction: ', pred
print 'acc: ', float(count) / float(len(label_o))

