from CNN import ChordCNN, get_accuracy, training_loop, load_model
import ChordDataset
from ChordDataset import get_input
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datetime import datetime
import os


if __name__ == "__main__":
    model = ChordCNN()

    dataset = ChordDataset.ChordDataset("data/")
    chords = {"0": "C", "1": "G", "2": "D"}

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

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
                epochs = input("Please enter number of epochs: ")
                training_loop(epochs, model, train_loader, criterion, optimizer)
                torch.save(model.state_dict(), datetime.now().strftime("weights-%Y-%m-%d_%H-%M-%S.pt"))
                get_accuracy(model, test_loader)
            case "2":
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