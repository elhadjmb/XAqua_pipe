"""import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from engine import train_one_epoch, evaluate  # Import these from torchvision references
from mrcnn.dataset import CocoDataset

# Initialize the dataset
rootd = r"/home/hadjm/PycharmProjects/AquaCounter/mrcnn/test_ds/"
transform = transforms.Compose([transforms.ToTensor()])

full_dataset = CocoDataset(root=rootd, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# all_ids = full_dataset.ids
# train_ids = all_ids[:train_size]
# val_ids = all_ids[train_size:]

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# # Create the actual training and validation datasets
# train_dataset = DatasetHandler(rootd, ids=train_ids, transform=transform)
# val_dataset = DatasetHandler(rootd, ids=val_ids, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model
model = maskrcnn_resnet50_fpn(pretrained=False)

# Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Initialize the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Number of epochs
num_epochs = 11

# Move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print("Starting training")
# Training loop
for epoch in range(num_epochs):
    # Train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    print("Trained one epoch")
    # Update the learning rate
    lr_scheduler.step()

    # Evaluate on the validation dataset (assuming you have a valid_loader)
    evaluate(model, valid_loader, device=device)
    print("Evaluated on validation set")

# Save the model
torch.save(model.state_dict(), rootd + "/mrcnn_fpn_resnet50.pth")



def training_sp(model, epoch, num_epochs, optimizer, data_loader_train, device, evaluate):
    model.train()

    preds_formatted = []  # Store all predictions here
    gt_formatted = []  # Store all ground truths here

    tot_loss_box_reg = 0
    tot_loss_classifier = 0
    tot_loss_objectness = 0
    tot_loss_rpn_box_reg = 0
    total_loss = 0
    msg(f"Epoch {epoch + 1} / {num_epochs}")
    for imgs, annotations in tqdm(data_loader_train):
        try:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()}
                           for t in annotations]

            optimizer.zero_grad()

            loss_dict = model(imgs, annotations)
            loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            tot_loss_box_reg += loss_box_reg
            tot_loss_classifier += loss_classifier
            tot_loss_objectness += loss_objectness
            tot_loss_rpn_box_reg += loss_rpn_box_reg
            total_loss += losses.item()

            losses.backward()
            optimizer.step()

            if evaluate:
                preds = get_preds(model, imgs)
                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)


        except Exception as e:
            raise Exception(
                f"Error occurred during training in epoch {epoch + 1}. Error: {e}")

    avg_train_loss = total_loss / len(data_loader_train)
    msg(f"Training Loss: {avg_train_loss}")
    avg_train_loss_box_reg = tot_loss_box_reg / len(data_loader_train)
    avg_train_loss_classifier = tot_loss_classifier / len(data_loader_train)
    avg_train_loss_objectness = tot_loss_objectness / len(data_loader_train)
    avg_train_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_train)

    return model, avg_train_loss, preds_formatted, gt_formatted, avg_train_loss_box_reg, avg_train_loss_classifier, avg_train_loss_objectness, avg_train_loss_rpn_box_reg



def validation_sp(model, data_loader_val, epoch, device):
    model.eval()
    total_val_loss = 0

    preds_formatted = []  # Store all predictions here
    gt_formatted = []  # Store all ground truths here
    tot_loss_box_reg = 0
    tot_loss_classifier = 0
    tot_loss_objectness = 0
    tot_loss_rpn_box_reg = 0
    with torch.no_grad():  # Turn off gradients for validation
        for imgs, annotations in tqdm(data_loader_val):
            try:
                imgs = list(img.to(device) for img in imgs)
                annotations = [
                    {k: v.to(device) for k, v in t.items()} for t in annotations]

                # Generate predictions
                preds = model(imgs)

                # Calculate validation loss
                loss_dict = get_loss_dict(model, imgs, annotations)
                loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
                tot_loss_box_reg += loss_box_reg
                tot_loss_classifier += loss_classifier
                tot_loss_objectness += loss_objectness
                tot_loss_rpn_box_reg += loss_rpn_box_reg
                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)

            except Exception as e:
                raise Exception(
                    f"Error occurred during validation in epoch {epoch + 1}. Error: {e}")
    avg_val_loss = total_val_loss / len(data_loader_val)
    msg(f"Validation Loss: {avg_val_loss}")
    avg_val_loss_box_reg = tot_loss_box_reg / len(data_loader_val)
    avg_val_loss_classifier = tot_loss_classifier / len(data_loader_val)
    avg_val_loss_objectness = tot_loss_objectness / len(data_loader_val)
    avg_val_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_val)
    return model, avg_val_loss, preds_formatted, gt_formatted, avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness, avg_val_loss_rpn_box_reg




fff = [os.path.join(root_path, f) for f in os.listdir(root_path) if
                     (f.endswith(".jpg") or f.endswith(".png"))]
    random.shuffle(fff)
    for ff in fff:
        MRCNN.inference(ff, categories=dataset.categories)

    input("CONTINUE?????????????????????")"""