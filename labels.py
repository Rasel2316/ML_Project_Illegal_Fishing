import pandas as pd
import joblib

class ModelPredictor:
    def __init__(self, model_path, test_data_path, output_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_path = output_path

    def load_model(self):
        # Load the trained model from the joblib file
        self.loaded_model = joblib.load(self.model_path)

    def load_test_data(self):
        # Load your test data
        self.test_data = pd.read_csv(self.test_data_path)

    def predict_and_save(self):
        # Apply the loaded model to generate labels for the test data
        predicted_labels = self.loaded_model.predict(self.test_data)

        # Save the predicted labels to a CSV file
        pd.DataFrame(predicted_labels, columns=['predicted_labels']).to_csv(self.output_path, index=False)

        print(f"Predicted labels saved to '{self.output_path}'")

if __name__ == "__main__":
    # Replace these paths with the actual paths
    model_path =r"C:\\Users\\Rasel Mondal\Downloads\\ML Project\\rf_SelectKBest_model.joblib"
    test_data_path = r'C:\\Users\\Rasel Mondal\Downloads\\ML Project\\test_imputed.csv'
    output_path = r'C:\\Users\\Rasel Mondal\Downloads\\ML Project\\predicted.csv'

    predictor = ModelPredictor(model_path, test_data_path, output_path)
    predictor.load_model()
    predictor.load_test_data()
    predictor.predict_and_save()
