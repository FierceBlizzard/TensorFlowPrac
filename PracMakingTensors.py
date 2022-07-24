import tensorflow as tf

#the rank of a tensor is the demsion 
#one string would be a rank 1
tensor1 = tf.Variable([["this is a string"], ["applepie"]], tf.string)

#changing shape
#tensor2 = tf.reshape(tensor1, [2, 3, 1]) error, invalid size

#the amount of elements is figures out by multiplying everything. So this is 5^4
t = tf.zeros([5,5,5,5])
print(t)
t = tf.reshape(t, 625)
print(t)

#if we aren't good at math, we can put -1 and tensorflow will guess what the shape should be
t = tf.reshape(t, [125, -1])
print(t)
#types of tensors
#Variable
#Constant
#Placeholder
#SparseTensor

#Except for Variable, all other types of tensors are immutable 