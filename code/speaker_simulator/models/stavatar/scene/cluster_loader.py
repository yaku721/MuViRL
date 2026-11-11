from torch.utils.data import Subset, DataLoader

def create_clustered_dataloaders(camera_dataset, clustered_timestamps, num_workers=2):

    clustered_dataloaders = []
    for cluster_indices in clustered_timestamps:
        subset = Subset(camera_dataset, cluster_indices)
        loader = DataLoader(
            subset,
            batch_size=None,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        clustered_dataloaders.append(loader)

    return clustered_dataloaders
