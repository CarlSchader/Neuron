import numpy as np
import neuron



#   Training/Testing Binary Adder

net = neuron.neuralNet([3,16,16,2])

net.load("binaryAdder")

for i in xrange(10000):
    x = np.array([np.random.randint(0,2,size=3)])
    if np.sum(x) == 3:
        y = np.array([1,1])
    elif np.sum(x) == 2:
        y = np.array([1,0])
    elif np.sum(x) == 1:
        y = np.array([0,1])
    else:
        y = np.array([0,0])

    out = round(net.forward(x)[0])
    print "input ", x[0]
    print "output ", out
    print "label ", y
    print
    print
    net.backward(y)
    if net.trainSize == 5:
        net.learn()

net.save("binaryAdder")
count = float(0)
correct = float(0)

net.load("binaryAdder")
# Testing
for i in xrange(2000):
   x = np.array([np.random.randint(0,2,size=3)])
   if np.sum(x) == 3:
       y = np.array([1,1])
   elif np.sum(x) == 2:
       y = np.array([1,0])
   elif np.sum(x) == 1:
       y = np.array([0,1])
   else:
       y = np.array([0,0])
   out = round(net.forward(x)[0])
   count += 1
   correct += 1
   for j in xrange(2):
       if out[j] != y[j]:
           correct -= 1
   print "Predic: ",round(net.forward(x)[0])
   print "Actual: ",y
   print
print "Correct: ", correct,'/',count,' ', (correct/count)*100, '%'






from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

net = ne.neuralNet([64,37,10])


train = input("Train or Test (True for train/False for test): ")

#   Training image Recognizer
if train == True:
    net.load("image")

    N = 1000

    for i in xrange(N):
        for k in xrange(len(x_train)):
            X = np.array([x_train[k]])
            Y = np.zeros(10)
            Y[y_train[k]] = 1
            net.forward(X)
            # print('X:',X)
            net.backward(Y)
            # print('Y:',Y)
            if net.trainSize == 5:
                net.learn()
        print float(i)/(N/100), "% Trained"
    print
    print
    print "Finished Training!"
    print
    print

    net.save("image")

#   Testing Image Recognizer
else:
    net.load("image")
    count = float(0)
    correct = float(0)

    for i in xrange(len(y_test)):
        Y = np.array(np.zeros(10))
        Y[y_test[i]] = 1
        X = np.array([x_test[i]])
        print y_test[i]
        out = round(net.forward(X)[0])
        print out
        print Y
        count += 1
        correct += 1
        for i in xrange(len(out)):
            if out[i] != Y[i]:
                correct -= 1


    print "Accuracy: ",correct,'/',count,(correct/count)*100, "%"






