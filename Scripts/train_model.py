
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve)
import matplotlib.pyplot as plt
import joblib

def build_ensemble_model(seq_input_shape: tuple, struct_input_shape: tuple):
    """Build CNN ensemble model combining sequence and structural features"""
    # Sequence branch (1D CNN)
    seq_input = layers.Input(shape=seq_input_shape, name='sequence_input')
    x = layers.Conv1D(64, 3, activation='relu')(seq_input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    
    # Structural branch (2D CNN)
    struct_input = layers.Input(shape=struct_input_shape, name='structure_input')
    y = layers.Conv2D(32, (3,3), activation='relu')(struct_input)
    y = layers.MaxPool2D((2,2))(y)
    y = layers.Conv2D(64, (3,3), activation='relu')(y)
    y = layers.GlobalMaxPool2D()(y)
    
    # Combined model
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[seq_input, struct_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def load_data(embeddings_path: str, structure_path: str):
    """Load preprocessed data"""
    sequence_data = pd.read_parquet(embeddings_path)
    structure_data = pd.read_csv(structure_path)
    
    # Align indices and extract labels
    labels = sequence_data.pop('label').values
    sequence_features = sequence_data.values
    structure_features = structure_data.drop(columns=['label']).values
    
    # Reshape for CNN input
    sequence_features = np.expand_dims(sequence_features, axis=-1)
    structure_features = structure_features.reshape(
        (-1, 64, 64, 1)  # Adjust based on your structural feature dimensions
    )
    
    return (sequence_features, structure_features), labels

def plot_training_history(history, output_dir: str):
    """Visualize training metrics"""
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png')
    plt.close()

def evaluate_model(model, X_test, y_test, output_dir: str):
    """Generate evaluation metrics and visualizations"""
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Classification report
    report = classification_report(y_test, y_pred_class, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f'{output_dir}/classification_report.csv')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train virulence prediction model')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to sequence embeddings file')
    parser.add_argument('--structure', type=str, required=True,
                       help='Path to structural features file')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for saved model')
    args = parser.parse_args()

    # Load and split data
    (X_seq, X_struct), y = load_data(args.embeddings, args.structure)
    X_train, X_val, y_train, y_val = train_test_split(
        [X_seq, X_struct], y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build model
    model = build_ensemble_model(
        seq_input_shape=X_seq.shape[1:],
        struct_input_shape=X_struct.shape[1:]
    )
    
    # Training callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        f'{args.output}/best_model.h5', save_best_only=True
    )
    
    # Handle class imbalance
    class_weights = {0: 1., 1: len(y_train)/sum(y_train)}  # Adjust weights for imbalance
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint]
    )
    
    # Save final model and training history
    model.save(f'{args.output}/final_model.h5')
    joblib.dump(history.history, f'{args.output}/training_history.pkl')
    
    # Generate evaluation plots
    plot_training_history(history, args.output)
    evaluate_model(model, X_val, y_val, args.output)

if __name__ == '__main__':
    main()