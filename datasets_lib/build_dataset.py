from functools import partial
from torch.utils.data import DataLoader

def get_dataset(args, processor):
    
    if args.experiment == 'code_gen_ft':
        from .Code_gen_data_loader import V_COT_SMART101_Dataset, img_train_collator_fn, img_test_collator_fn
        print('\n*****Load Train DataLoader*****')
        train_dataset = V_COT_SMART101_Dataset(args, 'train')
        valid_dataset = V_COT_SMART101_Dataset(args, 'valid')
        if args.mode == 'test':
            collator = partial(img_test_collator_fn, args=args, processor=processor, device='cuda')
        else:
            collator = partial(img_train_collator_fn, args=args, processor=processor, device='cuda')
        print('Code Generation FT')
        
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
        
        if args.mode == 'train':
            return train_loader, valid_loader
        else:
            return valid_loader    
        
    from .Code_Schema_data_loader import V_COT_SMART101_Dataset, img_dcp_train_collator_fn, img_dcp_test_collator_fn, img_train_collator_fn, img_test_collator_fn
    if args.mode == 'train':
        
        print('\n*****Load Train DataLoader*****')
        train_dataset = V_COT_SMART101_Dataset(args, 'train')
        valid_dataset = V_COT_SMART101_Dataset(args, 'valid')
        
        if args.use_img:
            collator = partial(img_train_collator_fn, args=args, processor=processor, device='cuda')
            print('Use Image')
        else:
            collator = partial(img_dcp_train_collator_fn, args=args, processor=processor, device='cuda')
            print('None use Image')
     
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
    
    elif args.mode == 'test':
        
        print('\n*****Load Test DataLoader*****')
        test_dataset = V_COT_SMART101_Dataset(args, args.mode)
        
        if args.use_img:
            collator = partial(img_test_collator_fn, args=args, processor=processor, device='cuda')
            print('Use Image')
        else:
            collator = partial(img_dcp_test_collator_fn, args=args, processor=processor, device='cuda')
            print('None use Image')
     
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return test_loader