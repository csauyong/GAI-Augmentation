import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K

# Step 1: Data Exploration and Preprocessing
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale the Time and Amount features
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Extract the minority class from the training set
    X_train_minority = X_train[y_train == 1]

    return X_train, X_test, y_train, y_test, X_train_minority

# Step 2: Implement the Generative Model (VAE)
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, intermediate_dim, latent_dim):
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(input_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # Loss function
    reconstruction_loss = binary_crossentropy(inputs, outputs) * input_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae

# Step 3: Train the Generative Model
def train_vae(vae, x_train_minority, batch_size, epochs):
    # Compile the VAE
    vae.compile(optimizer=Adam(learning_rate=0.001), metrics=None)

    # Train the VAE on the minority class
    vae.fit(x_train_minority, x_train_minority,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1)

# Step 4: Evaluate the Generated Data
def generate_data(vae, n_samples, latent_dim):
    z_samples = np.random.normal(0, 1, size=(n_samples, latent_dim))
    generated_data = vae.get_layer('decoder')(z_samples)
    return K.eval(generated_data)

def plot_tsne(real_data, generated_data):
    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack((real_data, generated_data))
    tsne_results = tsne.fit_transform(combined_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:len(real_data), 0], tsne_results[:len(real_data), 1], c='b', label='Real Data')
    plt.scatter(tsne_results[len(real_data):, 0], tsne_results[len(real_data):, 1], c='r', label='Generated Data')
    plt.legend()
    plt.title('t-SNE Visualization of Real and Generated Data')
    plt.show()
    
def evaluate_generated_data(x_train_minority, vae, n_samples, latent_dim):
    # Generate synthetic data points
    generated_data = generate_data(vae, n_samples, latent_dim)

    # Visualize the real and generated data using t-SNE
    plot_tsne(x_train_minority, generated_data)

    return x_train_minority, generated_data

# Step 5: Augment the Dataset
def build_classifier(input_dim):
    classifier = Sequential()
    classifier.add(Dense(128, input_dim=input_dim, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return classifier

def augment_training_set(X_train, y_train, generated_data):
    X_train_augmented = np.vstack((X_train, generated_data))
    y_train_augmented = np.hstack((y_train, np.ones(generated_data.shape[0])))
    return X_train_augmented, y_train_augmented

# Step 6: Train and Evaluate Classification Models
def train_classifier(classifier, X_train_augmented, y_train_augmented, batch_size, epochs):
    classifier.fit(X_train_augmented, y_train_augmented,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_split=0.1)
    
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test).round()
    y_prob = classifier.predict_proba(X_test)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC score
    print("ROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))

# Step 7: Compare Performance
def train_baseline_classifier(X_train, y_train, input_dim, batch_size, epochs):
    baseline_classifier = build_classifier(input_dim)
    baseline_classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)
    return baseline_classifier

def compare_classifiers(baseline_classifier, augmented_classifier, X_test, y_test):
    print("Baseline Classifier Performance:")
    evaluate_classifier(baseline_classifier, X_test, y_test)

    print("\nAugmented Classifier Performance:")
    evaluate_classifier(augmented_classifier, X_test, y_test)

# Main workflow
def main():
    # Load data and preprocess
    X_train, X_test, y_train, y_test, X_train_minority = load_and_preprocess_data()

    # Build and train the VAE
    latent_dim = 100
    vae, encoder, decoder = build_vae(X_train_minority.shape[1], latent_dim)
    batch_size = 32
    epochs = 100
    train_vae(vae, X_train_minority, batch_size, epochs)

    # Evaluate the generated data
    n_samples = len(X_train_minority)
    real_data, generated_data = evaluate_generated_data(X_train_minority, vae, n_samples, latent_dim)

    # Build and train the classifier with augmented data
    classifier = build_classifier(X_train.shape[1])
    X_train_augmented, y_train_augmented = augment_training_set(X_train, y_train, generated_data)
    train_classifier(classifier, X_train_augmented, y_train_augmented, batch_size, epochs)

    # Evaluate classifier performance
    evaluate_classifier(classifier, X_test, y_test)

    # Train a baseline classifier and compare the performance
    baseline_classifier = train_baseline_classifier(X_train, y_train, X_train.shape[1], batch_size, epochs)
    compare_classifiers(baseline_classifier, classifier, X_test, y_test)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
