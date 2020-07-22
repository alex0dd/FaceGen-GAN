import tensorflow as tf
from . import GAN_Wrapper

def ema_update(model, model_ema, beta=0.9999):
    """
    Performs a model update by using exponential moving average (EMA) 
    of first model's weights to update second model's weights 
    as defined in "The Unusual Effectiveness of Averaging in GAN Training" (https://arxiv.org/abs/1806.04498), 
    realizing the following update:
    
    model_ema.weights = beta * model_ema.weights + (1 - beta) * model.weights
    
    :param model: original, gradient descent trained model.
    :param model_ema: clone of original model, that will get updated using EMA.
    :param beta: EMA update weight, determines what portion of model_ema weights should be kept (default=0.9999).
    """
    # for each model layer index
    for i in range(len(model.layers)):
        updating_weights = model.layers[i].get_weights() # original model's weights
        ema_old_weights = model_ema.layers[i].get_weights() # ema model's weights
        ema_new_weights = [] # ema model's update weights
        if len(updating_weights) != len(ema_old_weights):
            # weight list length mismatch between model's weights and ema model's weights
            print("Different weight length")
            # copy ema weights directly from the model's weights
            ema_new_weights = updating_weights
        else:
            # for each weight tensor of original model's weights list
            for j in range(len(updating_weights)):
                n_weight = beta * ema_old_weights[j] + (1 - beta) * updating_weights[j]
                ema_new_weights.append(n_weight)
        # update weights
        model_ema.layers[i].set_weights(ema_new_weights)

class GAN_WrapperEMA(GAN_Wrapper):

    def __init__(self, discriminator, generator, generator_ema, beta=0.9999):
        super(GAN_WrapperEMA, self).__init__(discriminator, generator)
        self.generator_ema = generator_ema
        self.generator_ema.set_weights(self.generator.get_weights())

        self.beta = beta

        self.current_step = tf.Variable(initial_value=0, trainable=False)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN_WrapperEMA, self).compile(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn)

    def ema_step(self):
        ema_update(self.generator, self.generator_ema, beta=self.beta)

    def call(self, x, training=False):
        """
        This method is overridden only because it is required by tf.keras.Model
        """
        pass

    @tf.function
    def train_step(self, data):
        stats_dict = super(GAN_WrapperEMA, self).train_step(data)
        self.current_step.assign_add(1)
        return stats_dict
