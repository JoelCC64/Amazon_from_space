import torch
import numpy as np
from sklearn.metrics import f1_score

def entrenar_modelo(modelo, data_loader, criterion, optimizer, device, epochs=2):
    """
    To run the basic training loop over the complete dataset.
    """
    modelo.train()

    for epoch in range(epochs):
        running_loss = 0.0
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, (imagenes, targets) in enumerate(data_loader):
            imagenes = imagenes.to(device)
            targets = targets.to(device)

            # To reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = modelo(imagenes)

            # To calculate the loss
            loss = criterion(outputs, targets)

            # Backward propagation
            loss.backward()

            # To update the weights
            optimizer.step()

            running_loss += loss.item()

            # To print progress every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"Batch [{batch_idx}/{len(data_loader)}] - Current Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(data_loader)
        print(f"==> End of Epoch {epoch + 1} - Global Average Loss: {epoch_loss:.4f} <==")

def entrenamiento_validacion(model, train_loader, val_loader, criterion, optimizer, device, epochs=2, threshold=0.5):
    """
    To run the training loop, calculate validation loss, and evaluate the local validation F1-score 
    (using 'samples' average) at the end of each epoch.
    """
    for epoch in range(epochs):
        # Training phase
        model.train()
        accumulative_train_loss = 0.0
        print(f'\n --- Training epoch {epoch + 1}/{epochs} ---')
        for batch_idx, (imagenes, tags) in enumerate(train_loader):
            imagenes, tags = imagenes.to(device), tags.to(device)

            optimizer.zero_grad()
            outputs = model(imagenes)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            accumulative_train_loss += loss.item()
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}] - Train Loss: {loss.item():.4f}')

        avg_train_loss = accumulative_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        accumulative_eval_loss = 0.0

        all_predictions = []
        all_tags = []

        with torch.no_grad():
            for image, tags in val_loader:
                image, tags = image.to(device), tags.to(device)
    
                output = model(image)
                loss = criterion(output, tags)
                accumulative_eval_loss += loss.item()
    
                probabilidad_eval = torch.sigmoid(output)
                indices_predichos = (probabilidad_eval > threshold).int()
    
                # To store predictions and ground truth labels
                all_predictions.append(indices_predichos.cpu().numpy())
                all_tags.append(tags.cpu().numpy())
                
        avg_eval_loss = accumulative_eval_loss / len(val_loader)

        # To concatenate results from all batches
        all_predictions = np.vstack(all_predictions)
        all_tags = np.vstack(all_tags)

        # To calculate F1-score
        score_f1 = f1_score(all_tags, all_predictions, average='samples')

        print(f"\n--> EPOCH {epoch + 1} SUMMARY <--")
        print(f"Average Train Loss      : {avg_train_loss:.4f}")
        print(f"Average Validation Loss : {avg_eval_loss:.4f}")
        print(f"Validation F1-Score     : {score_f1:.4f}")
