from functools import partial
from torch.utils.data import DataLoader
from .Code_Schema_data_loader import V_COT_SMART101_Dataset, img_dcp_train_collator_fn, img_dcp_test_collator_fn

def get_dataset(args, processor):
    if args.mode == 'train':
        
        print('\n*****Load Train DataLoader*****')
        train_dataset = V_COT_SMART101_Dataset(args, 'train')
        valid_dataset = V_COT_SMART101_Dataset(args, 'valid')
        
        collator = partial(img_dcp_train_collator_fn, args=args, processor=processor, device='cuda')
     
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return train_loader, valid_loader
    
    elif args.mode in ['supervised_test', 'zeroshot_test']:
        
        print('\n*****Load Test DataLoader*****')
        test_dataset = V_COT_SMART101_Dataset(args, args.mode)
        
        collator = partial(img_dcp_test_collator_fn, args=args, processor=processor, device='cuda')
     
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return test_loader