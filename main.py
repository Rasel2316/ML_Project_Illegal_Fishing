from classify import Classification  # Assuming the class is defined in a file named 'classification.py'

def main():
    # Example usage
    classifier = Classification(path='fishing_imputed.csv', clf_opt='dt', no_of_selected_features=5)
    classifier.classification()

if __name__ == "__main__":
    main()
