from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class DataGenerator(DataLoader):
    
    def get_classes_to_idx(self):
        return self.dataset.dataset.class_to_idx
    
    def get_classes(self):
        return self.dataset.dataset.classes
        
def create_data_generators(dataset_name, domain, data_path = "data", batch_size = 16, 
                      transformations = transforms.ToTensor(), num_workers = 1, split_ratios = [0.8, 0.1, 0.1]):
    """
    Args:
        dataset_name (string)
        domain (string) - valid domain of the dataset dataset_name
        data_path (string) - valid path, which contains dataset_name folder 
        batch_size (int)
        transformations (callable) - optional transform to be applied on an image sample
        num_workers (int) - multi-process data loading 
        split_ratios (list of ints, len(split_ratios) = 3) - ratios of train, validation and test parts
        
    Return:
        3 data generators  - for train, validation and test data
        
    """
    
    dataset = create_dataset(dataset_name, domain, data_path, transformations)
    
    len_dataset = len(dataset)
    train_size = int(len_dataset * split_ratios[0])
    val_size = int(len_dataset * split_ratios[1])
    test_size = len_dataset - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataGenerator(train_dataset, batch_size = batch_size, 
                                  shuffle=True, num_workers=num_workers)
    
    val_dataloader = DataGenerator(val_dataset, batch_size = batch_size, 
                                shuffle=False, num_workers=num_workers)
    
    test_dataloader = DataGenerator(test_dataset, batch_size = batch_size, 
                                 shuffle=False, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader
    

    
def create_dataset(dataset_name, domain, data_path, transformations):
    """
    Args:
        dataset_name (string)
        domain (string) - valid domain of the dataset dataset_name
        data_path (string) - valid path, which contains dataset_name folder 
        transformations (callable) - optional transform to be applied on an image sample
        
    Return:
        torchvision.dataset object 
        
    """
    
    assert dataset_name in ["office-31"], f"Dataset {dataset_name} is not implemented"
    
    if dataset_name == "office-31":
        
        dataset_domains = ["amazon", "dslr", "webcam"]
        
        assert domain in dataset_domains, f"Incorrect domain {domain}: dataset {dataset_name} domains: {dataset_domains}"
        
        dataset = ImageFolder(f"{data_path}/{dataset_name}/{domain}/images", transform=transformations)
        
    return dataset