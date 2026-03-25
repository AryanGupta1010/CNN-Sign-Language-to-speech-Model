from load_dataset import load_dataset,split_grouped_data
from model import build_model
import tensorflow as tf
import numpy as np



X_train,y_train,X_val,y_val=split_grouped_data(load_dataset("dataset"),0.2)

model=build_model(num_classes=2)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history=model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=35,
    batch_size=5,
    callbacks=[early_stop]
)

preds = model.predict(X_val)
pred_labels = np.argmax(preds, axis=1)

print("Predictions:", pred_labels)
print("True labels:", y_val)


model.save("gesture_model.keras")