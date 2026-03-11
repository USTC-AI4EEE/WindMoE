from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Windpower

DATASET_REGISTRY = {
    'goldwind': Dataset_Windpower,
    'jilin': Dataset_Windpower,
    'fujian': Dataset_Windpower,
}

def data_provider(args, use: str):
    
    dataset_name = getattr(args, 'dataset', 'goldwind')
    DatasetClass = DATASET_REGISTRY.get(dataset_name)

    if not DatasetClass:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please register it in data_factory.py")

    if dataset_name == 'goldwind':
        index_col = 'dtime'
    else:  
        index_col = 'DateTime'

    dataset = DatasetClass(
        data_path=args.data_path,
        dataset_name=args.dataset,
        station=args.station,
        weather=args.weather,
        use=use,
        input_length=args.input_length,
        output_length=args.output_length,
        index_col=index_col,
    )
    
    print(f"Data for '{use}' on station {args.station}: {len(dataset)} samples")
    
    if len(dataset) == 0:
        return dataset, None
    
    shuffle_flag = 'train' in use
    drop_last_flag = 'train' in use

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=drop_last_flag
    )
    
    return dataset, data_loader