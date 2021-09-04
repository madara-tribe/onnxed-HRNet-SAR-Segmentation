from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def run_tta(model, X_test, tta_step=3, bs=1):
    test_datagen = ImageDataGenerator(horizontal_flip=True,
                                      #vertical_flip=True,
                                      fill_mode='constant')
    tta_steps = tta_step
    predictions = []
    for i in range(tta_steps):
        preds = model.predict(test_datagen.flow(X_test, batch_size=bs, shuffle=False), steps = len(X_test)/bs)
        predictions.append(preds)
    pred = np.mean(predictions, axis=0)
    return pred
