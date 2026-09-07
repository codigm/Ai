import os
import zipfile

def main():
    zip_filename = "kaggle_dataset.zip"
    
    # Files needed for SFT training on Kaggle
    files_to_include = [
        "model_chat_base.pt",    # The brain to start training from
        "chat_dataset.jsonl",    # The new 52k Alpaca dataset
        "tokenizer.json",        # The vocabulary
    ]
    
    # Add all python scripts
    for file in os.listdir("."):
        if file.endswith(".py"):
            files_to_include.append(file)
            
    print(f"Creating {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                print(f"Adding {file}...")
                zipf.write(file)
            else:
                print(f"Warning: {file} not found!")
                
    print(f"\n✅ Success! {zip_filename} has been created.")
    print(f"Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    print("\nYou can now upload this single zip file directly to Kaggle!")

if __name__ == "__main__":
    main()
