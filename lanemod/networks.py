import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, UpSampling2D

class Encoder(keras.layers.Layer):
    def __init__(self, inch, outch):
        super().__init__()
        self.conv1 = Conv2D((inch + outch)//2, 3, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(outch, 3, padding='same')
        self.bn2 = BatchNormalization()

        self.mp = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')

        self.conv3 = Conv2D(outch, 3, padding='same')
        self.bn3 = BatchNormalization()

    def call(self, input_tensor, training=True, *args, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.mp(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        return x

        
class Decoder(keras.layers.Layer):
    def __init__(self, inch, outch):
        super().__init__()
        self.conv1 = Conv2D((inch + outch)//2, 3, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(outch, 3, padding='same')
        self.bn2 = BatchNormalization()

        self.up = UpSampling2D(size=(2,2))

        self.conv3 = Conv2D(outch, 3, padding='same')
        self.bn3 = BatchNormalization()

    def call(self, input_tensor, *args, training=True, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.up(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        
        return x

class Residual(keras.layers.Layer):
    def __init__(self, inch, hich):
        super().__init__()
        self.conv1 = Conv2D(hich, 3, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(inch, 3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, input_tensor, *args, training=True, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)

class Aggregation(keras.layers.Layer):
    def __init__(self, inch, outch, hich=None):
        if hich is None:
            hich = outch
        super().__init__()
        self.conv1 = Conv2D(hich, 3, padding='same')
        self.bn1 = BatchNormalization()
        self.up = UpSampling2D(size=(2,2))
        self.conv2 = Conv2D(outch, 3, padding='same')
        self.bn2 = BatchNormalization()

    def call(self, input_tensor, other_tensor, *args, training=True, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x+self.up(other_tensor))
        x = self.bn2(x, training=training)
        # x += input_tensor
        return tf.nn.relu(x)

class LaneNetBig(keras.Model):
    def __init__(self):
        super().__init__()
        self.Encoder1 = Encoder(3, 32)        # input channel : 3,  output channel : 32
        self.Encoder2 = Encoder(32, 64)       # input channel : 32, output channel : 64
        self.Encoder3 = Encoder(64, 128)      # input channel : 64,  output channel : 128
        self.Residual1 = Residual(128, 256)   # input channel : 128  output channel : 256
        self.Residual2 = Residual(128, 256)   # input channel : 128,  output channel : 256
        self.Decoder1 = Decoder(128, 64)      # input channel : 128,  output channel : 64
        self.Decoder2 = Decoder(64, 32)       # input channel : 64,  output channel : 32
        self.Decoder3 = Decoder(32, 1)        # input channel : 32,  output channel : 1

        self.Aggregation1 = Aggregation(3, 32)
        self.Aggregation2 = Aggregation(32, 64)
        self.Aggregation3 = Aggregation(64, 128)
        self.Aggregation4 = Aggregation(128, 128)
        self.Aggregation5 = Aggregation(128, 64)
        self.Aggregation6 = Aggregation(64, 32)
        self.Aggregation7 = Aggregation(32, 1, hich=32)
        
        self.AggregationA = Aggregation(32, 64)
        self.AggregationB = Aggregation(64, 128)
        self.AggregationC = Aggregation(128, 128)
        self.AggregationD = Aggregation(128, 64)
        self.AggregationE = Aggregation(64, 32, hich=64)

        self.Aggregation_a = Aggregation(64, 128)
        self.Aggregation_b = Aggregation(128, 128)
        self.Aggregation_c = Aggregation(128, 64, hich = 128)


    def call(self, input_tensor, *args, training=True, **kwargs):
        e1o = self.Encoder1(input_tensor, training=training) 

        e2o = self.Encoder2(e1o, training=training)

        e3o = self.Encoder3(e2o, training=training) 

        r1o = self.Residual1(e3o, training=training) 

        r2o = self.Residual2(r1o, training=training) 

        d1o = self.Decoder1(r2o, training=training) 

        d2o = self.Decoder2(d1o, training=training) 

        d3o = self.Decoder3(d2o, training=training) 

        a1o = self.Aggregation1(input_tensor, e1o)

        b1o = self.AggregationA(e1o, e2o, training=training)

        a2o = self.Aggregation2(a1o, b1o, training=training)

        c1o = self.Aggregation_a(e2o, e3o, training=training)

        b2o = self.AggregationB(b1o, c1o, training=training)

        a3o = self.Aggregation3(a2o, b2o, training=training)

        c2o = self.Aggregation_b(c1o, r1o, training=training)

        b3o = self.AggregationC(b2o, c2o, training=training)

        a4o = self.Aggregation4(a3o, b3o, training=training)

        c3o = self.Aggregation_c(c2o, r2o, training=training)

        b4o = self.AggregationD(b3o, c3o, training=training)

        a5o = self.Aggregation5(a4o, b4o, training=training)

        b5o = self.AggregationE(b4o, d1o, training=training)

        a6o = self.Aggregation6(a5o, b5o, training=training)

        a7o = self.Aggregation7(a6o, d2o, training=training)

        return tf.math.sigmoid(d3o + a7o)

class LaneNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.e1 = Encoder(3, 32)
        self.e2 = Encoder(32, 64)
        self.r1 = Residual(64, 128)
        self.r2 = Residual(64, 128)
        self.d1 = Decoder(64, 32)
        self.d2 = Decoder(32, 1)
        self.a1 = Aggregation(3, 32)
        self.a2 = Aggregation(32, 64)
        self.a3 = Aggregation(64, 64)
        self.a4 = Aggregation(64, 32)
        self.a5 = Aggregation(32, 1, hich=32)
        
        self.b1 = Aggregation(32, 64)
        self.b2 = Aggregation(64, 64)
        self.b3 = Aggregation(64, 32, hich=64)


    def call(self, input_tensor, *args, training=True, **kwargs):
        e1o = self.e1(input_tensor, training=training) 
        e2o = self.e2(e1o, training=training) 
        r1o = self.r1(e2o, training=training) 
        r2o = self.r2(r1o, training=training) 
        d1o = self.d1(r2o, training=training) 
        d2o = self.d2(d1o, training=training) 

        a1o = self.a1(input_tensor, e1o)

        b1o = self.b1(e1o, e2o, training=training)
        a2o = self.a2(a1o, b1o, training=training)

        b2o = self.b2(b1o, r1o, training=training)
        a3o = self.a3(a2o, b2o, training=training)

        b3o = self.b3(b2o, r2o, training=training)
        a4o = self.a4(a3o, b3o, training=training)

        a5o = self.a5(a4o, d1o, training=training)

        return tf.math.sigmoid(d2o + a5o)
