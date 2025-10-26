"""Transfer learning training script using MobileNetV2.
Usage:
python training/train.py --data-dir /path/to/data --epochs 10 --batch-size 16 --output models/best_model.h5
"""
import argparse, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model(input_shape=(224,224,3), num_classes=2):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   horizontal_flip=True, rotation_range=15)
    train_flow = train_gen.flow_from_directory(args.data_dir, target_size=(224,224),
                                               batch_size=args.batch_size, subset='training')
    val_flow = train_gen.flow_from_directory(args.data_dir, target_size=(224,224),
                                             batch_size=args.batch_size, subset='validation')
    num_classes = train_flow.num_classes
    model = build_model(num_classes=num_classes)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ckpt = ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy')
    early = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(train_flow, validation_data=val_flow, epochs=args.epochs, callbacks=[ckpt, early])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', default='api/models/best_model.h5')
    args = p.parse_args()
    main(args)
