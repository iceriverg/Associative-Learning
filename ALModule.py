import tensorflow as tf
import numpy as np


class ALModule:
  def __init__(self, labeledBatch, labels, unlabeledBatch):
    self.labeledBatch = labeledBatch
    self.labels = labels

    # Set up the classCount map
    classCount = {}
    for el in labels:
      if el in classCount : classCount[el] = classCount[el] + 1
      else classCount[el] = 1
    self.classCount = classCount

    self.unlabeledBatch = unlabeledBatch
    
  def computePab(self):
    # m : batchsize of A
    # n : batchsize of B
    # output dimension = m X n

    # First, compute similarity matrix
    simMat = tf.matmul(self.labeledBatch, tf.transpose(self.unlabeldBatch))
    
    # Perform row-wise softmax
    return tf.nn.softmax(simMat, axis=1)

  def computePba(self):
    # m : batchsize of A
    # n : batchsize of B
    # output dimension = n X m
    return tf.matmul(self.unlabeledBatch, tf.transpose(self.labeldBatch))

    # First, compute similarity matrix
    simMat = tf.matmul(self.labeledBatch, tf.transpose(self.unlabeldBatch))
    
    # Perform row-wise softmax
    return tf.nn.softmax(simMat, axis=1)

  def computePaba(self):
    # m : batchsize of A
    # n : batchsize of B
    # Dimension of Pab : m X n
    # Dimension of Pba : n X m
    # output dimension = m X m
    return tf.matmul(self.computePab(),self.computePba())
  
  def getCorrectRoundTrip(self):
    # 1/labeledBatchSize if class(i) == class(j)
    # 0 otherwise
    output = []
    for el1 in labels:
      for el2 in labels:
        sublist = []
        if el1 == el2 : sublist.append(1.0/float(classCount[el]))
        else : sublist.append(0.0)
        output.append(sublist)

    return np.asarray(output) 

  def getWalkerLoss(self):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.computePaba(), labels=self.getCorrectRoundTrip()))

  def getVisitLoss(self):

    # Pvisit dimension : [1,unlabeled batch count]
    pVisit = tf.reduce_mean(self.computePab())
    BCount = len(self.unlabeledBatch)

    # Uniform target distribution V
    V = np.ones(Bcount)/Bcount
    return tf.nn.softmax_cross_entropy_with_logits(logits=pVisit, labels=V)
