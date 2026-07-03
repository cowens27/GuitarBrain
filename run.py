from CNN import ChordCNN, get_accuracy, training_loop, load_model, initialize_dataset
from ChordDataset import get_input
import torch
from datetime import datetime
import os

if __name__ == "__main__":
    model = ChordCNN()
    chords = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G"}

    train_loader = None
    test_loader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    while True:
        print("1) Train")
        print("2) Test")
        print("3) Load Model")
        print("4) Use")
        print("5) Quit")

        choice = input("Please enter a choice: ")

        match choice:
            case "1":
                if not model.dataset_loaded:
                    train_loader, test_loader = initialize_dataset()
                    model.dataset_loaded = True
                epochs = input("Please enter number of epochs: ")
                training_loop(epochs, model, train_loader, criterion, optimizer)
                torch.save(model.state_dict(), datetime.now().strftime("2-0-weights-%Y-%m-%d_%H-%M-%S.pt"))
                get_accuracy(model, test_loader)
            case "2":
                if not model.dataset_loaded:
                    train_loader, test_loader = initialize_dataset()
                    model.dataset_loaded = True
                get_accuracy(model, test_loader)
            case "3":
                load_model(model)
            case "4":
                path = input("Please enter path: ")
                if path == "":
                    print("Model failed to load. Incomplete path.")
                if not os.path.exists(path):
                    print("Model failed to load. File does not exist.")
                else:
                    model.eval()
                    input_spec = get_input(path)
                    input_spec = input_spec.unsqueeze(0)
                    prediction = model.forward(input_spec)
                    prediction = prediction.argmax()
                    prediction = int(prediction)

                    print("Prediction:", chords[str(prediction)])
            case "4":
                break