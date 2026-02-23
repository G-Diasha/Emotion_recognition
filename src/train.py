from src.model import build_model
from src.dataset import load_datasets, prepare_dataset

train_ds, val_ds = load_datasets()
train_ds = prepare_dataset(train_ds)
val_ds = prepare_dataset(val_ds)

#build model
model = build_model()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)
model.save("models/emotion_resnet50.keras")
