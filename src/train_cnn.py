import tensorflow as tf
from src.get_data import get_face_data_label_origin, get_emotion_type

batch_size = 128
num_steps = 2000
learning_rate = 0.001
# Network Parameters
num_classes = 7  # total classes
dropout = 0.25  # Dropout, probability to keep units
train_x, train_y, test_x, test_y, valid_x, valid_y = get_face_data_label_origin()
emotion_dict = get_emotion_type()


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']
        x = tf.cast(x, tf.float32)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 42, 42, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Convolution Layer with 128 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


model = tf.estimator.Estimator(model_fn)
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_x}, y=train_y,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
print(model.train(input_fn, steps=num_steps))

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': valid_x}, y=valid_y,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
print(model.evaluate(input_fn))


# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_x}, shuffle=False)
# Use the model to predict the images class
preds = list(model.predict(input_fn))
yes = 0

# Display
appear_counter = {i: {j: 0 for j in range(7)} for i in range(7)}
total_counter = {i: 0 for i in range(7)}

for i in range(len(test_x)):
    # plt.imshow(np.reshape(test_images[i], [48, 48]), cmap='gray')
    # plt.show()
    # count appearance
    tmp_count = total_counter.get(test_y[i])
    total_counter.__setitem__(test_y[i], tmp_count + 1)
    tmp_class = appear_counter.get(test_y[i])
    tmp_count = tmp_class.get(preds[i])
    appear_counter.get(test_y[i]).__setitem__(preds[i], tmp_count + 1)
    if preds[i] == test_y[i]:
        yes += 1

print(yes / len(test_x))
for i in range(7):
    appear = appear_counter.get(i)
    total = total_counter.get(i)
    rate_list = []
    for j in range(7):
        appear_count = appear.get(j)
        total_count = total
        rate_list.append(appear_count / total_count * 100)

    print("| {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% |".format(
        emotion_dict.get(i),
        rate_list[0],
        rate_list[1],
        rate_list[2],
        rate_list[3],
        rate_list[4],
        rate_list[5],
        rate_list[6],
    ))