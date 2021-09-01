from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(
    filepath="/content/drive/MyDrive/RohithWorkspace/models/ResUNet-segModel-weights.hdf5",
    verbose=1,
    save_best_only=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.0001, factor=0.2
)
