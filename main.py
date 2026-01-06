from preproccessing import *

try:
        train = load_split("training")
        val = load_split("validation")
        test = load_split("test")
        print(f"Training set size:   {len(train)}")
        print(f"Validation set size: {len(val)}")
        print(f"Test set size:       {len(test)}")
        
        if train:
            print("Sample training item keys:", train[0].keys())
            print("Sample text snippet:", train[0]["text"][:100])
            
    except Exception as e:
        print(f"Error: {e}")