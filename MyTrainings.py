import tensorflow as tf

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

        for images in train_loader:
            with tf.GradientTape() as tape:
                outputs = model(images)
                loss = loss_fn(images, outputs)  # Reconstruction loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss(loss)
            epoch_accuracy(images, outputs)

        loss_values.append(epoch_loss.result().numpy())
        accuracy_values.append(epoch_accuracy.result().numpy())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_values[-1]:.4f}, Accuracy: {accuracy_values[-1]:.4f}")

    print("Training complete!")
    return loss_values, accuracy_values


