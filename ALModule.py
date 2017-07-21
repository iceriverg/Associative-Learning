import tensorflow as tf
import numpy as np


class ALModule:
    def __init__(self, labeledBatch, labels, unlabeledBatch, unlabeledBatchCount):
        self.labeledBatch = labeledBatch
        self.labels = labels
        self.unlabeledBatch = unlabeledBatch
        self.unlabeledBatchCount = unlabeledBatchCount
    
    def computePab(self):
        # m : batchsize of A
        # n : batchsize of B
        # output dimension = m X n

        # First, compute similarity matrix
        simMat = tf.matmul(self.labeledBatch, tf.transpose(self.unlabeledBatch))
    
        # Perform row-wise softmax
        return tf.nn.softmax(simMat)

    def computePba(self):
        # m : batchsize of A
        # n : batchsize of B
        # output dimension = n X m

        # First, compute similarity matrix
        simMat = tf.matmul(self.unlabeledBatch, tf.transpose(self.labeledBatch))
    
        # Perform row-wise softmax
        return tf.nn.softmax(simMat)

    def computePaba(self):
        # m : batchsize of A
        # n : batchsize of B
        # Dimension of Pab : m X n
        # Dimension of Pba : n X m
        # output dimension = m X m
        return tf.matmul(self.computePab(),self.computePba())
  
    def getCorrectRoundTrip(self):
        # 1/|class(i)| if class(i) == class(j)
        # 0 otherwise
        equalMat = tf.cast(tf.equal(tf.reshape(self.labels, [-1,1]), tf.reshape(self.labels, [-1])), tf.float32)
        return equalMat / tf.reduce_sum(equalMat,0) 

    def getWalkerLoss(self):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.computePaba(), labels=self.getCorrectRoundTrip()))

    def getVisitLoss(self):
        # Pvisit dimension : [1,unlabeled batch count]
        pVisit = tf.reduce_mean(self.computePab(), 0)
        
        # Uniform target distribution V
        V = tf.ones([self.unlabeledBatchCount])/tf.cast(self.unlabeledBatchCount, tf.float32)
        
        return tf.nn.softmax_cross_entropy_with_logits(logits=pVisit, labels=V)
